#include "nim_cosmos.h"

#include <windows.h>
#include <winhttp.h>

#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <algorithm>
#include <iostream>

// stb_image_write — implementation compiled in stb_write_impl.cpp
#include "stb_image_write.h"
// stb_image — for decoding the response frame
#include "stb_image.h"

// ─────────────────────────────────────────────────────────────────────────────
//  Base-64 encode / decode
// ─────────────────────────────────────────────────────────────────────────────

static std::string b64_encode(const uint8_t* data, size_t len)
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

static std::vector<uint8_t> b64_decode(const std::string& encoded)
{
    static const int D[256] = {
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,62,-1,-1,-1,63,
        52,53,54,55,56,57,58,59,60,61,-1,-1,-1,-1,-1,-1,
        -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,
        15,16,17,18,19,20,21,22,23,24,25,-1,-1,-1,-1,-1,
        -1,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,
        41,42,43,44,45,46,47,48,49,50,51,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
        -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,
    };
    std::vector<uint8_t> out;
    out.reserve(encoded.size() * 3 / 4);
    uint32_t buf = 0;
    int bits = 0;
    for (unsigned char c : encoded) {
        int v = D[c];
        if (v < 0) continue;
        buf = (buf << 6) | (uint32_t)v;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((uint8_t)(buf >> bits));
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Minimal MP4/H.264 wrapper for a single JPEG frame
// ─────────────────────────────────────────────────────────────────────────────
// Cosmos expects MP4 video input.  For a single frame we create a minimal
// MJPEG-in-MP4 container (ftyp + mdat + moov) that any decoder can read.
// This avoids pulling in a full video encoder.

static void write_u32be(std::vector<uint8_t>& v, uint32_t x)
{
    v.push_back((uint8_t)(x >> 24));
    v.push_back((uint8_t)(x >> 16));
    v.push_back((uint8_t)(x >>  8));
    v.push_back((uint8_t)(x >>  0));
}

static void write_tag(std::vector<uint8_t>& v, const char* tag)
{
    v.push_back((uint8_t)tag[0]);
    v.push_back((uint8_t)tag[1]);
    v.push_back((uint8_t)tag[2]);
    v.push_back((uint8_t)tag[3]);
}

// Build a minimal single-frame MJPEG MP4.
static std::vector<uint8_t> wrap_jpeg_as_mp4(
    const std::vector<uint8_t>& jpeg, int width, int height)
{
    // We'll build ftyp + mdat + moov atoms.
    // This is a bare-minimum MP4 that wraps one JPEG frame as MJPEG.
    std::vector<uint8_t> mp4;
    mp4.reserve(jpeg.size() + 1024);

    // ── ftyp ─────────────────────────────────────────────────────────────────
    {
        size_t start = mp4.size();
        write_u32be(mp4, 0); // placeholder size
        write_tag(mp4, "ftyp");
        write_tag(mp4, "isom");
        write_u32be(mp4, 0x200); // minor version
        write_tag(mp4, "isom");
        write_tag(mp4, "iso2");
        // patch size
        uint32_t sz = (uint32_t)(mp4.size() - start);
        mp4[start+0] = (uint8_t)(sz >> 24);
        mp4[start+1] = (uint8_t)(sz >> 16);
        mp4[start+2] = (uint8_t)(sz >>  8);
        mp4[start+3] = (uint8_t)(sz >>  0);
    }

    // ── mdat ─────────────────────────────────────────────────────────────────
    size_t mdat_offset = mp4.size();
    {
        uint32_t mdat_size = (uint32_t)(8 + jpeg.size());
        write_u32be(mp4, mdat_size);
        write_tag(mp4, "mdat");
        mp4.insert(mp4.end(), jpeg.begin(), jpeg.end());
    }

    // ── moov (simplified — one video track, one sample) ──────────────────────
    // This is the minimum set of atoms MP4 decoders need.
    auto write_box = [&](const char* tag, auto&& write_body) {
        size_t start = mp4.size();
        write_u32be(mp4, 0); // placeholder
        write_tag(mp4, tag);
        write_body();
        uint32_t sz = (uint32_t)(mp4.size() - start);
        mp4[start+0] = (uint8_t)(sz >> 24);
        mp4[start+1] = (uint8_t)(sz >> 16);
        mp4[start+2] = (uint8_t)(sz >>  8);
        mp4[start+3] = (uint8_t)(sz >>  0);
    };

    auto write_fullbox = [&](const char* tag, uint8_t ver, uint32_t flags, auto&& body) {
        write_box(tag, [&]() {
            mp4.push_back(ver);
            mp4.push_back((uint8_t)(flags >> 16));
            mp4.push_back((uint8_t)(flags >> 8));
            mp4.push_back((uint8_t)(flags));
            body();
        });
    };

    uint32_t sample_size = (uint32_t)jpeg.size();
    uint32_t data_offset = (uint32_t)(mdat_offset + 8);

    write_box("moov", [&]() {
        // mvhd
        write_fullbox("mvhd", 0, 0, [&]() {
            write_u32be(mp4, 0); // creation time
            write_u32be(mp4, 0); // modification time
            write_u32be(mp4, 24); // timescale (24 fps)
            write_u32be(mp4, 1);  // duration (1 tick = 1 frame)
            write_u32be(mp4, 0x00010000); // rate 1.0
            mp4.push_back(0x01); mp4.push_back(0x00); // volume 1.0
            for (int i = 0; i < 10; ++i) mp4.push_back(0); // reserved
            // identity matrix (9 x uint32)
            uint32_t matrix[] = {0x00010000,0,0,0,0x00010000,0,0,0,0x40000000};
            for (auto m : matrix) write_u32be(mp4, m);
            for (int i = 0; i < 24; ++i) mp4.push_back(0); // pre-defined
            write_u32be(mp4, 2); // next track id
        });

        // trak
        write_box("trak", [&]() {
            // tkhd
            write_fullbox("tkhd", 0, 3, [&]() {
                write_u32be(mp4, 0); write_u32be(mp4, 0); // creation/mod time
                write_u32be(mp4, 1); // track id
                write_u32be(mp4, 0); // reserved
                write_u32be(mp4, 1); // duration
                write_u32be(mp4, 0); write_u32be(mp4, 0); // reserved
                mp4.push_back(0); mp4.push_back(0); // layer
                mp4.push_back(0); mp4.push_back(0); // alt group
                mp4.push_back(0); mp4.push_back(0); // volume
                mp4.push_back(0); mp4.push_back(0); // reserved
                uint32_t matrix[] = {0x00010000,0,0,0,0x00010000,0,0,0,0x40000000};
                for (auto m : matrix) write_u32be(mp4, m);
                write_u32be(mp4, (uint32_t)width << 16);  // width fixed-point
                write_u32be(mp4, (uint32_t)height << 16); // height fixed-point
            });

            // mdia
            write_box("mdia", [&]() {
                // mdhd
                write_fullbox("mdhd", 0, 0, [&]() {
                    write_u32be(mp4, 0); write_u32be(mp4, 0);
                    write_u32be(mp4, 24); // timescale
                    write_u32be(mp4, 1);  // duration
                    write_u32be(mp4, 0x55C40000); // language und
                });
                // hdlr
                write_fullbox("hdlr", 0, 0, [&]() {
                    write_u32be(mp4, 0); // pre-defined
                    write_tag(mp4, "vide");
                    write_u32be(mp4, 0); write_u32be(mp4, 0); write_u32be(mp4, 0);
                    const char* name = "VideoHandler";
                    for (const char* p = name; *p; ++p) mp4.push_back(*p);
                    mp4.push_back(0);
                });
                // minf
                write_box("minf", [&]() {
                    // vmhd
                    write_fullbox("vmhd", 0, 1, [&]() {
                        write_u32be(mp4, 0); write_u32be(mp4, 0);
                    });
                    // dinf/dref
                    write_box("dinf", [&]() {
                        write_fullbox("dref", 0, 0, [&]() {
                            write_u32be(mp4, 1); // entry count
                            write_fullbox("url ", 0, 1, [&]() {}); // self-contained
                        });
                    });
                    // stbl
                    write_box("stbl", [&]() {
                        // stsd — sample description (MJPEG)
                        write_fullbox("stsd", 0, 0, [&]() {
                            write_u32be(mp4, 1); // entry count
                            // mjpa sample entry
                            size_t entry_start = mp4.size();
                            write_u32be(mp4, 0); // placeholder size
                            write_tag(mp4, "mjpa");
                            for (int i = 0; i < 6; ++i) mp4.push_back(0); // reserved
                            mp4.push_back(0); mp4.push_back(1); // data ref index
                            for (int i = 0; i < 16; ++i) mp4.push_back(0); // pre-defined + reserved
                            mp4.push_back((uint8_t)(width >> 8));
                            mp4.push_back((uint8_t)(width));
                            mp4.push_back((uint8_t)(height >> 8));
                            mp4.push_back((uint8_t)(height));
                            write_u32be(mp4, 0x00480000); // horiz res 72 dpi
                            write_u32be(mp4, 0x00480000); // vert res 72 dpi
                            write_u32be(mp4, 0); // data size
                            mp4.push_back(0); mp4.push_back(1); // frame count
                            for (int i = 0; i < 32; ++i) mp4.push_back(0); // compressor name
                            mp4.push_back(0); mp4.push_back(0x18); // depth 24
                            mp4.push_back(0xFF); mp4.push_back(0xFF); // pre-defined -1
                            uint32_t esz = (uint32_t)(mp4.size() - entry_start);
                            mp4[entry_start+0] = (uint8_t)(esz >> 24);
                            mp4[entry_start+1] = (uint8_t)(esz >> 16);
                            mp4[entry_start+2] = (uint8_t)(esz >>  8);
                            mp4[entry_start+3] = (uint8_t)(esz >>  0);
                        });
                        // stts — time to sample
                        write_fullbox("stts", 0, 0, [&]() {
                            write_u32be(mp4, 1); // entry count
                            write_u32be(mp4, 1); // sample count
                            write_u32be(mp4, 1); // sample delta
                        });
                        // stsc — sample to chunk
                        write_fullbox("stsc", 0, 0, [&]() {
                            write_u32be(mp4, 1);
                            write_u32be(mp4, 1); // first chunk
                            write_u32be(mp4, 1); // samples per chunk
                            write_u32be(mp4, 1); // sample desc index
                        });
                        // stsz — sample size
                        write_fullbox("stsz", 0, 0, [&]() {
                            write_u32be(mp4, 0); // uniform size (0 = per-sample)
                            write_u32be(mp4, 1); // sample count
                            write_u32be(mp4, sample_size);
                        });
                        // stco — chunk offset
                        write_fullbox("stco", 0, 0, [&]() {
                            write_u32be(mp4, 1);
                            write_u32be(mp4, data_offset);
                        });
                    }); // stbl
                }); // minf
            }); // mdia
        }); // trak
    }); // moov

    return mp4;
}

// ─────────────────────────────────────────────────────────────────────────────
//  JPEG encode from RGBA8 buffer
// ─────────────────────────────────────────────────────────────────────────────

static void jpeg_cb(void* ctx, void* data, int size)
{
    auto* buf = static_cast<std::vector<uint8_t>*>(ctx);
    const uint8_t* p = static_cast<const uint8_t*>(data);
    buf->insert(buf->end(), p, p + size);
}

static std::vector<uint8_t> rgba8_to_jpeg(const uint8_t* rgba, int w, int h, int quality = 85)
{
    // Convert RGBA8 → RGB8
    std::vector<uint8_t> rgb(w * h * 3);
    for (int i = 0; i < w * h; ++i) {
        rgb[i * 3 + 0] = rgba[i * 4 + 0];
        rgb[i * 3 + 1] = rgba[i * 4 + 1];
        rgb[i * 3 + 2] = rgba[i * 4 + 2];
    }
    std::vector<uint8_t> jpeg;
    jpeg.reserve(w * h);
    stbi_write_jpg_to_func(jpeg_cb, &jpeg, w, h, 3, rgb.data(), quality);
    return jpeg;
}

// ─────────────────────────────────────────────────────────────────────────────
//  WinHTTP POST (reused from nim_vlm pattern)
// ─────────────────────────────────────────────────────────────────────────────

static std::string http_post(const char* host, int port,
                              const char* path,
                              const std::string& body,
                              const char* content_type,
                              bool use_https,
                              const char* api_key,
                              int timeout_ms = 120000) // 2 min for AI generation
{
    auto to_wide = [](const char* s) -> std::wstring {
        if (!s || !*s) return {};
        int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
        std::wstring w(n, 0);
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
        return w;
    };

    std::string result;

    HINTERNET hSession = WinHttpOpen(L"Cosmos-Transfer/1.0",
                                     WINHTTP_ACCESS_TYPE_NO_PROXY,
                                     WINHTTP_NO_PROXY_NAME,
                                     WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return "ERROR:WinHttpOpen failed";

    WinHttpSetTimeouts(hSession, timeout_ms, timeout_ms, timeout_ms, timeout_ms);

    HINTERNET hConnect = WinHttpConnect(hSession, to_wide(host).c_str(),
                                         (INTERNET_PORT)port, 0);
    if (!hConnect) {
        WinHttpCloseHandle(hSession);
        return "ERROR:WinHttpConnect failed";
    }

    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"POST",
                                             to_wide(path).c_str(),
                                             nullptr, WINHTTP_NO_REFERER,
                                             WINHTTP_DEFAULT_ACCEPT_TYPES,
                                             use_https ? WINHTTP_FLAG_SECURE : 0);
    if (!hRequest) {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return "ERROR:WinHttpOpenRequest failed";
    }

    std::wstring hdr = L"Content-Type: " + to_wide(content_type) + L"\r\n";
    WinHttpAddRequestHeaders(hRequest, hdr.c_str(), (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD);

    if (api_key && api_key[0]) {
        std::wstring auth = L"Authorization: Bearer " + to_wide(api_key) + L"\r\n";
        WinHttpAddRequestHeaders(hRequest, auth.c_str(), (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD);
    }

    // For large bodies, set content length explicitly
    std::wstring cl = L"Content-Length: " + std::to_wstring(body.size()) + L"\r\n";
    WinHttpAddRequestHeaders(hRequest, cl.c_str(), (DWORD)-1, WINHTTP_ADDREQ_FLAG_ADD);

    BOOL ok = WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                  (LPVOID)body.c_str(), (DWORD)body.size(),
                                  (DWORD)body.size(), 0);

    if (!ok || !WinHttpReceiveResponse(hRequest, nullptr)) {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return "ERROR:Request failed (is Cosmos reachable at " +
               std::string(host) + ":" + std::to_string(port) + "?)";
    }

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
//  Simple JSON string value extractor
// ─────────────────────────────────────────────────────────────────────────────

static std::string json_string_value(const std::string& json, const std::string& key)
{
    std::string needle = "\"" + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return {};
    pos += needle.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == ':')) ++pos;
    if (pos >= json.size() || json[pos] != '"') return {};
    ++pos;
    std::string val;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            ++pos;
            if (json[pos] == 'n') val += '\n';
            else if (json[pos] == '"') val += '"';
            else if (json[pos] == '\\') val += '\\';
            else val += json[pos];
        } else {
            val += json[pos];
        }
        ++pos;
    }
    return val;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Cosmos Transfer API call
// ─────────────────────────────────────────────────────────────────────────────

CosmosResult cosmos_transfer(
    const CosmosConfig& cfg,
    const uint8_t*      beauty_rgba8,
    const uint8_t*      depth_rgba8,
    const uint8_t*      seg_rgba8,
    int                 width,
    int                 height)
{
    CosmosResult res;

    // ── Encode each AOV as JPEG → MP4 → base64 ──────────────────────────────
    std::cout << "[cosmos] Encoding AOV buffers (" << width << "x" << height << ")...\n";

    auto encode_aov = [&](const uint8_t* rgba, const char* name) -> std::string {
        auto jpeg = rgba8_to_jpeg(rgba, width, height, 90);
        auto mp4  = wrap_jpeg_as_mp4(jpeg, width, height);
        std::cout << "[cosmos] " << name << ": JPEG=" << jpeg.size()
                  << " MP4=" << mp4.size() << " bytes\n";
        return b64_encode(mp4.data(), mp4.size());
    };

    std::string beauty_b64 = encode_aov(beauty_rgba8, "beauty");
    std::string depth_b64  = depth_rgba8 ? encode_aov(depth_rgba8, "depth") : "";
    std::string seg_b64    = seg_rgba8   ? encode_aov(seg_rgba8,   "seg")   : "";

    // ── Build JSON request ───────────────────────────────────────────────────
    // Escape prompt for JSON
    std::string prompt_escaped;
    for (char c : std::string(cfg.prompt)) {
        if (c == '"') prompt_escaped += "\\\"";
        else if (c == '\\') prompt_escaped += "\\\\";
        else if (c == '\n') prompt_escaped += "\\n";
        else prompt_escaped += c;
    }

    std::string json = "{\n";
    json += "  \"model\": \"" + std::string(cfg.model) + "\",\n";
    json += "  \"prompt\": \"" + prompt_escaped + "\",\n";
    json += "  \"video\": \"data:video/mp4;base64," + beauty_b64 + "\",\n";

    if (!depth_b64.empty()) {
        char w[32]; snprintf(w, sizeof(w), "%.2f", cfg.depth_weight);
        json += "  \"depth\": {\n";
        json += "    \"control_weight\": " + std::string(w) + ",\n";
        json += "    \"control\": \"data:video/mp4;base64," + depth_b64 + "\"\n";
        json += "  },\n";
    }

    if (!seg_b64.empty()) {
        char w[32]; snprintf(w, sizeof(w), "%.2f", cfg.seg_weight);
        json += "  \"seg\": {\n";
        json += "    \"control_weight\": " + std::string(w) + ",\n";
        json += "    \"control\": \"data:video/mp4;base64," + seg_b64 + "\"\n";
        json += "  },\n";
    }

    json += "  \"seed\": " + std::to_string(cfg.seed) + "\n";
    json += "}";

    std::cout << "[cosmos] Request JSON: " << json.size() << " bytes\n";
    std::cout << "[cosmos] Sending to " << cfg.host << ":" << cfg.port << "...\n";

    // ── POST to Cosmos API ───────────────────────────────────────────────────
    std::string endpoint = cfg.endpoint;
    if (endpoint.empty()) endpoint = "/v1/cv/nvidia/cosmos-transfer2p5-2b";

    std::string response = http_post(
        cfg.host, cfg.port, endpoint.c_str(),
        json, "application/json",
        cfg.use_https, cfg.api_key);

    if (response.empty()) {
        snprintf(res.error, sizeof(res.error), "Empty response from Cosmos");
        return res;
    }

    if (response.rfind("ERROR:", 0) == 0) {
        snprintf(res.error, sizeof(res.error), "%s", response.c_str());
        return res;
    }

    std::cout << "[cosmos] Response: " << response.size() << " bytes\n";

    // Log first 500 chars for debugging
    std::cout << "[cosmos] Response preview: "
              << response.substr(0, std::min((size_t)500, response.size())) << "\n";

    // ── Parse response ───────────────────────────────────────────────────────
    // Expected: { "b64_video": "<base64>" } or similar
    std::string video_b64 = json_string_value(response, "b64_video");
    if (video_b64.empty())
        video_b64 = json_string_value(response, "video");
    if (video_b64.empty())
        video_b64 = json_string_value(response, "output");

    if (video_b64.empty()) {
        // Check for error message
        std::string err = json_string_value(response, "error");
        if (err.empty()) err = json_string_value(response, "message");
        if (err.empty()) err = json_string_value(response, "detail");
        snprintf(res.error, sizeof(res.error), "No output in response: %.400s",
                 err.empty() ? response.c_str() : err.c_str());
        return res;
    }

    // Strip data URI prefix if present
    {
        size_t comma = video_b64.find(',');
        if (comma != std::string::npos && comma < 100)
            video_b64 = video_b64.substr(comma + 1);
    }

    // ── Decode the response video/image ──────────────────────────────────────
    std::vector<uint8_t> decoded = b64_decode(video_b64);
    std::cout << "[cosmos] Decoded output: " << decoded.size() << " bytes\n";

    if (decoded.empty()) {
        snprintf(res.error, sizeof(res.error), "Failed to decode base64 output");
        return res;
    }

    // Try to decode as image directly (JPEG/PNG)
    int img_w = 0, img_h = 0, img_ch = 0;
    uint8_t* img = stbi_load_from_memory(decoded.data(), (int)decoded.size(),
                                          &img_w, &img_h, &img_ch, 4);
    if (img) {
        res.success = true;
        res.pixels  = new uint8_t[img_w * img_h * 4];
        memcpy(res.pixels, img, img_w * img_h * 4);
        res.width   = img_w;
        res.height  = img_h;
        stbi_image_free(img);
        std::cout << "[cosmos] Decoded image: " << img_w << "x" << img_h << "\n";
        return res;
    }

    // If it's an MP4, try to find the JPEG frame inside the mdat atom
    for (size_t i = 0; i + 4 < decoded.size(); ++i) {
        if (decoded[i] == 0xFF && decoded[i+1] == 0xD8 &&
            decoded[i+2] == 0xFF) {
            // Found JPEG SOI marker
            img = stbi_load_from_memory(decoded.data() + i,
                                         (int)(decoded.size() - i),
                                         &img_w, &img_h, &img_ch, 4);
            if (img) {
                res.success = true;
                res.pixels  = new uint8_t[img_w * img_h * 4];
                memcpy(res.pixels, img, img_w * img_h * 4);
                res.width   = img_w;
                res.height  = img_h;
                stbi_image_free(img);
                std::cout << "[cosmos] Extracted JPEG from MP4: "
                          << img_w << "x" << img_h << "\n";
                return res;
            }
        }
    }

    snprintf(res.error, sizeof(res.error),
             "Could not decode output frame (%zu bytes)", decoded.size());
    return res;
}
