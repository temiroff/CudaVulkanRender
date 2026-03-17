#include "nim_vlm.h"

#include <windows.h>
#include <winhttp.h>

#include <string>
#include <vector>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <iostream>

// stb_image_write — implementation compiled in stb_write_impl.cpp
#include "stb_image_write.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Base-64 encoder
// ─────────────────────────────────────────────────────────────────────────────

static std::string base64_encode(const uint8_t* data, size_t len)
{
    static const char* T =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    for (size_t i = 0; i < len; i += 3) {
        uint32_t v = (uint32_t)(data[i]) << 16;
        if (i + 1 < len) v |= (uint32_t)(data[i + 1]) << 8;
        if (i + 2 < len) v |= (uint32_t)(data[i + 2]);
        out += T[(v >> 18) & 63];
        out += T[(v >> 12) & 63];
        out += (i + 1 < len) ? T[(v >> 6) & 63] : '=';
        out += (i + 2 < len) ? T[ v       & 63] : '=';
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Image preparation: downscale + ACES tone map + gamma → JPEG
// ─────────────────────────────────────────────────────────────────────────────

// ACES fitted (Narkowicz 2015) applied per-channel
static float aces_channel(float x)
{
    if (x <= 0.f) return 0.f;
    float v = (x * (2.51f * x + 0.03f)) / (x * (2.43f * x + 0.59f) + 0.14f);
    return std::max(0.f, std::min(1.f, v));
}

// Nearest-neighbour downscale of RGBA float4 → RGB uint8 JPEG buffer.
// max_dim: clamp longest side to this many pixels.
static void jpeg_write_cb(void* ctx, void* data, int size)
{
    auto* buf = static_cast<std::vector<uint8_t>*>(ctx);
    const uint8_t* p = static_cast<const uint8_t*>(data);
    buf->insert(buf->end(), p, p + size);
}

static std::vector<uint8_t> prepare_jpeg(
    const float* px,   // float4 linear HDR pixels
    int src_w, int src_h,
    int max_dim = 768, int quality = 75)
{
    // ── Compute output resolution (fit within max_dim × max_dim) ─────────────
    int dst_w = src_w, dst_h = src_h;
    if (dst_w > max_dim || dst_h > max_dim) {
        if (dst_w >= dst_h) {
            dst_h = (int)(dst_h * (float)max_dim / dst_w);
            dst_w = max_dim;
        } else {
            dst_w = (int)(dst_w * (float)max_dim / dst_h);
            dst_h = max_dim;
        }
        dst_w = std::max(1, dst_w);
        dst_h = std::max(1, dst_h);
    }

    // ── Nearest-neighbour resample + tone map + gamma ─────────────────────────
    std::vector<uint8_t> rgb(dst_w * dst_h * 3);
    for (int dy = 0; dy < dst_h; ++dy) {
        int sy = dy * src_h / dst_h;
        for (int dx = 0; dx < dst_w; ++dx) {
            int    sx = dx * src_w / dst_w;
            size_t si = ((size_t)sy * src_w + sx) * 4;  // float4 stride

            float r = aces_channel(px[si + 0]);
            float g = aces_channel(px[si + 1]);
            float b = aces_channel(px[si + 2]);

            // sRGB gamma ≈ 2.2
            r = std::powf(r, 1.f / 2.2f);
            g = std::powf(g, 1.f / 2.2f);
            b = std::powf(b, 1.f / 2.2f);

            size_t di = ((size_t)dy * dst_w + dx) * 3;
            rgb[di + 0] = (uint8_t)(r * 255.f + 0.5f);
            rgb[di + 1] = (uint8_t)(g * 255.f + 0.5f);
            rgb[di + 2] = (uint8_t)(b * 255.f + 0.5f);
        }
    }

    // ── JPEG encode ───────────────────────────────────────────────────────────
    std::vector<uint8_t> jpeg;
    jpeg.reserve(dst_w * dst_h);
    stbi_write_jpg_to_func(jpeg_write_cb, &jpeg, dst_w, dst_h, 3, rgb.data(), quality);
    return jpeg;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Simple JSON string-value extractor (handles basic escape sequences)
// ─────────────────────────────────────────────────────────────────────────────

static std::string json_string_value(const std::string& json, const std::string& key)
{
    std::string needle = "\"" + key + "\"";
    size_t pos = 0;
    while ((pos = json.find(needle, pos)) != std::string::npos) {
        pos += needle.size();
        // skip whitespace and colon
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' ||
               json[pos] == '\n' || json[pos] == '\r' || json[pos] == ':'))
            ++pos;
        if (pos >= json.size() || json[pos] != '"') continue;
        ++pos;  // skip opening quote
        std::string val;
        while (pos < json.size() && json[pos] != '"') {
            if (json[pos] == '\\' && pos + 1 < json.size()) {
                ++pos;
                switch (json[pos]) {
                    case 'n':  val += '\n'; break;
                    case 't':  val += '\t'; break;
                    case '"':  val += '"';  break;
                    case '\\': val += '\\'; break;
                    case 'r':  val += '\r'; break;
                    default:   val += json[pos]; break;
                }
            } else {
                val += json[pos];
            }
            ++pos;
        }
        return val;
    }
    return {};
}

// Extract the "content" field from an OpenAI-style chat completion response.
// The content may itself be a JSON string (nested).
static std::string extract_content(const std::string& response_json)
{
    return json_string_value(response_json, "content");
}

// ─────────────────────────────────────────────────────────────────────────────
//  WinHTTP helper — synchronous POST to http://host:port/path
// ─────────────────────────────────────────────────────────────────────────────

static std::string http_post(const char* host, int port,
                              const char* path,
                              const std::string& body,
                              const char* content_type,
                              bool use_https = false,
                              const char* api_key = nullptr,
                              int timeout_ms = 30000)
{
    // Convert narrow → wide
    auto to_wide = [](const char* s) -> std::wstring {
        if (!s || !*s) return {};
        int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
        std::wstring w(n, 0);
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
        return w;
    };

    std::string result;

    HINTERNET hSession = WinHttpOpen(L"NIM-VLM/1.0",
                                     WINHTTP_ACCESS_TYPE_NO_PROXY,
                                     WINHTTP_NO_PROXY_NAME,
                                     WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) { result = "ERROR:WinHttpOpen failed"; return result; }

    // Set timeouts
    WinHttpSetTimeouts(hSession, timeout_ms, timeout_ms, timeout_ms, timeout_ms);

    HINTERNET hConnect = WinHttpConnect(hSession,
                                         to_wide(host).c_str(),
                                         (INTERNET_PORT)port, 0);
    if (!hConnect) {
        WinHttpCloseHandle(hSession);
        result = "ERROR:WinHttpConnect failed";
        return result;
    }

    HINTERNET hRequest = WinHttpOpenRequest(hConnect,
                                             L"POST",
                                             to_wide(path).c_str(),
                                             nullptr,
                                             WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES,
                                             use_https ? WINHTTP_FLAG_SECURE : 0);
    if (!hRequest) {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        result = "ERROR:WinHttpOpenRequest failed";
        return result;
    }

    // Set Content-Type header
    std::wstring hdr = L"Content-Type: " + to_wide(content_type) + L"\r\n";
    WinHttpAddRequestHeaders(hRequest, hdr.c_str(), (DWORD)-1,
                              WINHTTP_ADDREQ_FLAG_ADD);

    // Set Authorization header when an API key is provided
    if (api_key && api_key[0] != '\0') {
        std::wstring auth = L"Authorization: Bearer " + to_wide(api_key) + L"\r\n";
        WinHttpAddRequestHeaders(hRequest, auth.c_str(), (DWORD)-1,
                                  WINHTTP_ADDREQ_FLAG_ADD);
    }

    BOOL ok = WinHttpSendRequest(hRequest,
                                  WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                  WINHTTP_NO_REQUEST_DATA, 0,
                                  (DWORD)body.size(), 0);
    if (ok)
        ok = WinHttpWriteData(hRequest,
                              body.c_str(), (DWORD)body.size(), nullptr);

    if (!ok || !WinHttpReceiveResponse(hRequest, nullptr)) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        result = "ERROR:Request/response failed (is NIM running at " +
                 std::string(host) + ":" + std::to_string(port) + "?)";
        return result;
    }

    // Read response in chunks
    DWORD bytes_avail = 0;
    do {
        WinHttpQueryDataAvailable(hRequest, &bytes_avail);
        if (bytes_avail == 0) break;
        std::vector<char> buf(bytes_avail + 1, 0);
        DWORD bytes_read = 0;
        if (WinHttpReadData(hRequest, buf.data(), bytes_avail, &bytes_read))
            result.append(buf.data(), bytes_read);
    } while (bytes_avail > 0);

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  JSON request builder
// ─────────────────────────────────────────────────────────────────────────────

// Escape a string for embedding in JSON (handles quotes and backslashes)
static std::string json_escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if      (c == '"')  out += "\\\"";
        else if (c == '\\') out += "\\\\";
        else if (c == '\n') out += "\\n";
        else if (c == '\r') out += "\\r";
        else                out += c;
    }
    return out;
}

static std::string recognition_prompt()
{
    return
        "You are a 3D asset cataloging assistant. Analyze these rendered views of a 3D model "
        "and identify the object. "
        "Respond ONLY with valid JSON — no markdown, no explanation — in exactly this format:\\n"
        "{\\\"category\\\": \\\"<category>\\\", "
        "\\\"object\\\": \\\"<specific object type>\\\", "
        "\\\"description\\\": \\\"<2-3 sentences describing visual appearance, materials, style>\\\", "
        "\\\"tags\\\": \\\"<8-12 comma-separated search tags>\\\"}\\n"
        "Categories: furniture, vehicle, architecture, nature, character, "
        "prop, interior, environment, abstract\\n"
        "Example: {\\\"category\\\": \\\"furniture\\\", "
        "\\\"object\\\": \\\"wooden chair\\\", "
        "\\\"description\\\": \\\"A rustic wooden chair with a high carved back and four legs. "
        "The seat is upholstered in dark leather with brass nail trim. "
        "Aged oak finish with visible wood grain.\\\", "
        "\\\"tags\\\": \\\"chair, wooden, furniture, seating, rustic, oak, antique, interior\\\"}";
}

static NimVlmResult parse_nim_response(const std::string& response)
{
    NimVlmResult result;

    if (response.substr(0, 6) == "ERROR:") {
        snprintf(result.error_msg, sizeof(result.error_msg), "%s", response.c_str() + 6);
        return result;
    }

    std::cout << "[NIM] Response: " << response.substr(0, 200) << "...\n";

    std::string content = extract_content(response);
    if (content.empty()) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "Could not find 'content' in response: %.200s", response.c_str());
        return result;
    }

    std::cout << "[NIM] Content: " << content << "\n";

    // Strip ```json ... ``` wrapper if present
    {
        size_t start = content.find('{');
        size_t end   = content.rfind('}');
        if (start != std::string::npos && end != std::string::npos && end > start)
            content = content.substr(start, end - start + 1);
    }

    std::string category    = json_string_value(content, "category");
    std::string object_str  = json_string_value(content, "object");
    std::string description = json_string_value(content, "description");
    std::string tags        = json_string_value(content, "tags");

    if (object_str.empty()) object_str = json_string_value(content, "object_type");

    if (category.empty() && object_str.empty()) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "VLM returned unparseable content: %.300s", content.c_str());
        return result;
    }

    if (!category.empty()) category[0] = (char)toupper((unsigned char)category[0]);

    snprintf(result.category,    sizeof(result.category),    "%s", category.c_str());
    snprintf(result.object_type, sizeof(result.object_type), "%s", object_str.c_str());
    snprintf(result.description, sizeof(result.description), "%s", description.c_str());
    snprintf(result.tags,        sizeof(result.tags),        "%s", tags.c_str());
    result.success = true;
    return result;
}

static std::string build_request_json(
    const char* model, float temperature, int max_tokens,
    const std::string& image_b64)
{
    const std::string prompt = recognition_prompt();

    // Format temperature with one decimal
    char temp_str[32];
    snprintf(temp_str, sizeof(temp_str), "%.2f", (double)temperature);

    std::string json =
        "{"
        "\"model\":\"" + std::string(model) + "\","
        "\"temperature\":" + temp_str + ","
        "\"max_tokens\":" + std::to_string(max_tokens) + ","
        "\"messages\":[{"
          "\"role\":\"user\","
          "\"content\":["
            "{"
              "\"type\":\"image_url\","
              "\"image_url\":{"
                "\"url\":\"data:image/jpeg;base64," + image_b64 + "\""
              "}"
            "},"
            "{"
              "\"type\":\"text\","
              "\"text\":\"" + prompt + "\""
            "}"
          "]"
        "}]"
        "}";
    return json;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Public API
// ─────────────────────────────────────────────────────────────────────────────

NimVlmResult nim_vlm_recognize(
    const NimVlmConfig& cfg,
    const float*        pixels_rgba_f32,
    int                 width,
    int                 height)
{
    NimVlmResult result;

    // ── Encode image ─────────────────────────────────────────────────────────
    auto jpeg = prepare_jpeg(pixels_rgba_f32, width, height);
    if (jpeg.empty()) {
        snprintf(result.error_msg, sizeof(result.error_msg),
                 "JPEG encoding failed");
        return result;
    }
    std::string b64 = base64_encode(jpeg.data(), jpeg.size());

    // ── Build and send request ────────────────────────────────────────────────
    std::string body = build_request_json(
        cfg.model, cfg.temperature, cfg.max_tokens, b64);

    std::cout << "[NIM] Sending " << (jpeg.size() / 1024) << " KB image ("
              << b64.size() << " B b64) to " << cfg.host << ":" << cfg.port << "\n";

    std::string response = http_post(
        cfg.host, cfg.port,
        "/v1/chat/completions",
        body, "application/json",
        cfg.use_https,
        cfg.api_key[0] ? cfg.api_key : nullptr);

    return parse_nim_response(response);
}

NimVlmResult nim_vlm_recognize_multi(
    const NimVlmConfig& cfg,
    const float* const* pixels,
    int                 count,
    int                 width,
    int                 height)
{
    NimVlmResult result;
    if (count <= 0 || !pixels) {
        snprintf(result.error_msg, sizeof(result.error_msg), "No images provided");
        return result;
    }

    // ── Encode all images to base64 JPEG ─────────────────────────────────────
    std::vector<std::string> b64_images;
    b64_images.reserve(count);
    for (int i = 0; i < count; ++i) {
        auto jpeg = prepare_jpeg(pixels[i], width, height);
        if (jpeg.empty()) {
            snprintf(result.error_msg, sizeof(result.error_msg), "JPEG encode failed for angle %d", i);
            return result;
        }
        b64_images.push_back(base64_encode(jpeg.data(), jpeg.size()));
    }

    // ── Build multi-image request JSON ────────────────────────────────────────
    char temp_str[32];
    snprintf(temp_str, sizeof(temp_str), "%.2f", (double)cfg.temperature);

    std::string content_arr;
    for (auto& b64 : b64_images) {
        content_arr +=
            "{"
              "\"type\":\"image_url\","
              "\"image_url\":{\"url\":\"data:image/jpeg;base64," + b64 + "\"}"
            "},";
    }
    // Append text prompt
    content_arr +=
        "{"
          "\"type\":\"text\","
          "\"text\":\"" + recognition_prompt() + "\""
        "}";

    std::string body =
        "{"
        "\"model\":\"" + std::string(cfg.model) + "\","
        "\"temperature\":" + temp_str + ","
        "\"max_tokens\":" + std::to_string(cfg.max_tokens) + ","
        "\"messages\":[{"
          "\"role\":\"user\","
          "\"content\":[" + content_arr + "]"
        "}]"
        "}";

    std::cout << "[NIM] Sending " << count << " angle images to "
              << cfg.host << ":" << cfg.port << "\n";

    std::string response = http_post(
        cfg.host, cfg.port,
        "/v1/chat/completions",
        body, "application/json",
        cfg.use_https,
        cfg.api_key[0] ? cfg.api_key : nullptr);

    return parse_nim_response(response);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Connection check — GET /v1/models (or just attempt to connect)
// ─────────────────────────────────────────────────────────────────────────────

bool nim_vlm_check_connection(const NimVlmConfig& cfg)
{
    auto to_wide = [](const char* s) -> std::wstring {
        if (!s || !*s) return {};
        int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
        std::wstring w(n, 0);
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
        return w;
    };

    HINTERNET hSession = WinHttpOpen(L"NIM-VLM-Ping/1.0",
                                      WINHTTP_ACCESS_TYPE_NO_PROXY,
                                      WINHTTP_NO_PROXY_NAME,
                                      WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return false;
    WinHttpSetTimeouts(hSession, 5000, 5000, 5000, 5000);

    HINTERNET hConnect = WinHttpConnect(hSession,
                                         to_wide(cfg.host).c_str(),
                                         (INTERNET_PORT)cfg.port, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); return false; }

    HINTERNET hReq = WinHttpOpenRequest(hConnect, L"GET", L"/v1/models",
                                         nullptr, WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES,
                                         cfg.use_https ? WINHTTP_FLAG_SECURE : 0);
    if (!hReq) {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }
    if (cfg.api_key[0]) {
        std::wstring auth = L"Authorization: Bearer " + to_wide(cfg.api_key) + L"\r\n";
        WinHttpAddRequestHeaders(hReq, auth.c_str(), (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD);
    }

    BOOL ok = WinHttpSendRequest(hReq, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                  WINHTTP_NO_REQUEST_DATA, 0, 0, 0)
           && WinHttpReceiveResponse(hReq, nullptr);

    DWORD status = 0;
    DWORD sz = sizeof(status);
    if (ok) {
        WinHttpQueryHeaders(hReq,
            WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
            WINHTTP_HEADER_NAME_BY_INDEX, &status, &sz,
            WINHTTP_NO_HEADER_INDEX);
    }

    WinHttpCloseHandle(hReq);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);

    // 200 OK or 401 Unauthorized both mean the server is reachable
    return ok && (status == 200 || status == 401 || status == 404);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Docker NIM launch
// ─────────────────────────────────────────────────────────────────────────────

bool nim_vlm_launch_docker(const NimVlmConfig& cfg, char* error_out, int error_len)
{
    if (error_out && error_len > 0) error_out[0] = '\0';

    // First check if Docker daemon is reachable
    int ping = system("docker info >nul 2>&1");
    if (ping != 0) {
        if (error_out)
            snprintf(error_out, (size_t)error_len,
                "Docker is not running. Please start Docker Desktop first.");
        return false;
    }

    // Authenticate with NGC registry if an API key is provided.
    // nvcr.io requires "docker login" before pulling — without this you get
    // "Access Denied" even though the image exists.
    if (cfg.api_key[0]) {
        char login_cmd[512];
        snprintf(login_cmd, sizeof(login_cmd),
            "docker login nvcr.io --username \"$oauthtoken\" --password \"%s\" >nul 2>&1",
            cfg.api_key);
        int login_rc = system(login_cmd);
        if (login_rc != 0 && error_out) {
            snprintf(error_out, (size_t)error_len,
                "NGC login failed. Check your API key at https://ngc.nvidia.com/setup/api-key");
            return false;
        }
    } else if (error_out) {
        snprintf(error_out, (size_t)error_len,
            "NGC API key is empty. Get one at https://ngc.nvidia.com/setup/api-key");
        return false;
    }

    // docker run -d --gpus all -p <port>:8000 \
    //   -e NGC_API_KEY=<key> \
    //   nvcr.io/nim/<model>:latest
    char cmd[2048];
    snprintf(cmd, sizeof(cmd),
        "start \"NIM\" cmd /k docker run --gpus all"
        " -p %d:8000"
        " -e NVIDIA_VISIBLE_DEVICES=all"
        " -e NGC_API_KEY=%s"
        " nvcr.io/nim/%s:latest",
        cfg.port,
        cfg.api_key,
        cfg.model);
    system(cmd);
    return true;
}
