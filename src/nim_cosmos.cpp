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
//  WinHTTP GET
// ─────────────────────────────────────────────────────────────────────────────

static std::string http_get(const char* host, int port, const char* path,
                             bool use_https, int timeout_ms = 300000)
{
    auto to_wide = [](const char* s) -> std::wstring {
        if (!s || !*s) return {};
        int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
        std::wstring w(n, 0);
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
        return w;
    };
    std::string result;
    HINTERNET hSession = WinHttpOpen(L"Cosmos/1.0", WINHTTP_ACCESS_TYPE_NO_PROXY,
                                     WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return "";
    WinHttpSetTimeouts(hSession, timeout_ms, timeout_ms, timeout_ms, timeout_ms);
    HINTERNET hConnect = WinHttpConnect(hSession, to_wide(host).c_str(), (INTERNET_PORT)port, 0);
    if (!hConnect) { WinHttpCloseHandle(hSession); return ""; }
    HINTERNET hReq = WinHttpOpenRequest(hConnect, L"GET", to_wide(path).c_str(),
                                         nullptr, WINHTTP_NO_REFERER,
                                         WINHTTP_DEFAULT_ACCEPT_TYPES,
                                         use_https ? WINHTTP_FLAG_SECURE : 0);
    if (!hReq) { WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession); return ""; }
    if (!WinHttpSendRequest(hReq, WINHTTP_NO_ADDITIONAL_HEADERS, 0, nullptr, 0, 0, 0) ||
        !WinHttpReceiveResponse(hReq, nullptr)) {
        WinHttpCloseHandle(hReq); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession);
        return "";
    }
    DWORD avail = 0;
    do {
        WinHttpQueryDataAvailable(hReq, &avail);
        if (!avail) break;
        std::vector<char> buf(avail + 1, 0);
        DWORD rd = 0;
        if (WinHttpReadData(hReq, buf.data(), avail, &rd)) result.append(buf.data(), rd);
    } while (avail > 0);
    WinHttpCloseHandle(hReq); WinHttpCloseHandle(hConnect); WinHttpCloseHandle(hSession);
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Multipart file upload via WinHTTP
// ─────────────────────────────────────────────────────────────────────────────

static std::string http_upload_file(const char* host, int port, const char* path,
                                     const char* filename,
                                     const uint8_t* data, size_t data_size,
                                     bool use_https)
{
    auto to_wide = [](const char* s) -> std::wstring {
        if (!s || !*s) return {};
        int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
        std::wstring w(n, 0);
        MultiByteToWideChar(CP_UTF8, 0, s, -1, w.data(), n);
        return w;
    };

    std::string boundary = "----CosmosUpload9f8e7d6c";
    std::string body;
    body += "--" + boundary + "\r\n";
    body += "Content-Disposition: form-data; name=\"files\"; filename=\"" + std::string(filename) + "\"\r\n";
    body += "Content-Type: application/octet-stream\r\n\r\n";
    body.append(reinterpret_cast<const char*>(data), data_size);
    body += "\r\n--" + boundary + "--\r\n";

    std::string ct = "multipart/form-data; boundary=" + boundary;
    return http_post(host, port, path, body, ct.c_str(), use_https, nullptr, 60000);
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
    const char* host = cfg.host;
    int port = cfg.port;
    bool https = cfg.use_https;

    // Helper: encode RGBA8 → JPEG → MP4, save to debug/, return MP4 bytes
    auto encode_and_save = [&](const uint8_t* rgba, const char* name) -> std::vector<uint8_t> {
        auto j = rgba8_to_jpeg(rgba, width, height, 90);
        auto m = wrap_jpeg_as_mp4(j, width, height);
        // Save to debug/ for inspection and easy resend
        std::string path = std::string("debug/cosmos_") + name + ".mp4";
        FILE* f = fopen(path.c_str(), "wb");
        if (f) { fwrite(m.data(), 1, m.size(), f); fclose(f); }
        std::cout << "[cosmos] " << name << " MP4: " << m.size() << " bytes → " << path << "\n";
        return m;
    };

    // ── 1. Encode AOVs as MP4 and save to debug/ ────────────────────────────
    std::cout << "[cosmos] Encoding AOVs (" << width << "x" << height << ")...\n";
    auto mp4 = encode_and_save(beauty_rgba8, "beauty");
    std::vector<uint8_t> depth_mp4, seg_mp4;
    if (depth_rgba8) depth_mp4 = encode_and_save(depth_rgba8, "depth");
    if (seg_rgba8)   seg_mp4   = encode_and_save(seg_rgba8,   "seg");

    // ── 2. Upload beauty MP4 to Gradio ──────────────────────────────────────
    std::cout << "[cosmos] Uploading to " << host << ":" << port << "...\n";
    std::string upload_resp = http_upload_file(
        host, port, "/gradio_api/upload",
        "beauty.mp4", mp4.data(), mp4.size(), https);
    std::cout << "[cosmos] Upload response: " << upload_resp << "\n";

    if (upload_resp.empty() || upload_resp[0] != '[') {
        snprintf(res.error, sizeof(res.error), "Upload failed: %.400s", upload_resp.c_str());
        return res;
    }
    // Parse server path from ["<path>"]
    std::string beauty_path;
    {
        size_t q1 = upload_resp.find('"');
        size_t q2 = upload_resp.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos)
            beauty_path = upload_resp.substr(q1 + 1, q2 - q1 - 1);
    }
    if (beauty_path.empty()) {
        snprintf(res.error, sizeof(res.error), "Failed to parse upload path");
        return res;
    }
    std::cout << "[cosmos] Uploaded to: " << beauty_path << "\n";

    // ── 3. Upload depth + seg control videos if provided ────────────────────
    std::string depth_path, seg_path;
    if (!depth_mp4.empty()) {
        std::string dr = http_upload_file(host, port, "/gradio_api/upload",
                                           "depth.mp4", depth_mp4.data(), depth_mp4.size(), https);
        size_t q1 = dr.find('"'), q2 = dr.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos)
            depth_path = dr.substr(q1 + 1, q2 - q1 - 1);
        std::cout << "[cosmos] Depth uploaded: " << depth_path << "\n";
    }
    if (!seg_mp4.empty()) {
        std::string sr = http_upload_file(host, port, "/gradio_api/upload",
                                           "seg.mp4", seg_mp4.data(), seg_mp4.size(), https);
        size_t q1 = sr.find('"'), q2 = sr.find('"', q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos)
            seg_path = sr.substr(q1 + 1, q2 - q1 - 1);
        std::cout << "[cosmos] Seg uploaded: " << seg_path << "\n";
    }

    // ── 4. Build generate_video request ─────────────────────────────────────
    // Escape prompt
    std::string prompt_esc;
    for (char c : std::string(cfg.prompt)) {
        if (c == '"') prompt_esc += "\\\"";
        else if (c == '\\') prompt_esc += "\\\\";
        else if (c == '\n') prompt_esc += "\\n";
        else prompt_esc += c;
    }

    // Inner JSON (the request passed as a string to generate_video)
    std::string inner = "{";
    inner += "\"name\":\"cosmos_render\",";
    inner += "\"prompt\":\"" + prompt_esc + "\",";
    inner += "\"video_path\":\"" + beauty_path + "\",";
    inner += "\"resolution\":\"480\",";
    inner += "\"num_steps\":20,";
    inner += "\"seed\":" + std::to_string(cfg.seed);
    if (!depth_path.empty()) {
        char w[32]; snprintf(w, sizeof(w), "%.2f", cfg.depth_weight);
        inner += ",\"depth\":{\"control_path\":\"" + depth_path + "\",\"control_weight\":" + w + "}";
    }
    if (!seg_path.empty()) {
        char w[32]; snprintf(w, sizeof(w), "%.2f", cfg.seg_weight);
        inner += ",\"seg\":{\"control_path\":\"" + seg_path + "\",\"control_weight\":" + w + "}";
    }
    inner += "}";

    // Outer JSON wrapping for Gradio API: {"data":["<inner_json_string>"]}
    // The inner JSON must be escaped as a string inside the outer JSON
    std::string inner_escaped;
    for (char c : inner) {
        if (c == '"') inner_escaped += "\\\"";
        else if (c == '\\') inner_escaped += "\\\\";
        else inner_escaped += c;
    }
    std::string outer = "{\"data\":[\"" + inner_escaped + "\"]}";

    std::cout << "[cosmos] Calling generate_video (" << outer.size() << " bytes)...\n";

    // ── 5. POST to /gradio_api/call/generate_video ──────────────────────────
    std::string call_resp = http_post(host, port,
        "/gradio_api/call/generate_video",
        outer, "application/json", https, nullptr, 10000);
    std::cout << "[cosmos] Call response: " << call_resp << "\n";

    // Parse event_id from {"event_id":"<id>"}
    std::string event_id = json_string_value(call_resp, "event_id");
    if (event_id.empty()) {
        snprintf(res.error, sizeof(res.error), "No event_id: %.400s", call_resp.c_str());
        return res;
    }

    // ── 6. Poll for result via SSE GET ──────────────────────────────────────
    std::string poll_path = "/gradio_api/call/generate_video/" + event_id;
    std::cout << "[cosmos] Polling " << poll_path << "...\n";

    // Gradio returns SSE: "event: complete\ndata: [...]"
    // The GET blocks until complete (up to timeout)
    std::string sse = http_get(host, port, poll_path.c_str(), https, 600000); // 10min timeout
    std::cout << "[cosmos] SSE response: " << sse.size() << " bytes\n";
    std::cout << "[cosmos] SSE preview: "
              << sse.substr(0, std::min((size_t)500, sse.size())) << "\n";

    if (sse.empty()) {
        snprintf(res.error, sizeof(res.error), "Empty SSE response (timeout?)");
        return res;
    }

    // Parse the "data:" line — contains JSON array
    // Look for video file path in the response
    // Format: event: complete\ndata: [{"video":{"path":"/tmp/...mp4",...}}, "result_json"]
    std::string video_url;
    {
        // Find "path" in the response
        size_t pos = sse.find("\"path\"");
        if (pos != std::string::npos) {
            pos = sse.find('"', pos + 6);  // skip "path"
            if (pos != std::string::npos) {
                pos = sse.find('"', pos + 1); // skip :
                // skip any whitespace/colon
                while (pos < sse.size() && (sse[pos] == ':' || sse[pos] == ' ' || sse[pos] == '"')) pos++;
                if (pos > 0) pos--; // back to opening quote
                size_t q1 = sse.find('"', pos);
                size_t q2 = sse.find('"', q1 + 1);
                if (q1 != std::string::npos && q2 != std::string::npos)
                    video_url = sse.substr(q1 + 1, q2 - q1 - 1);
            }
        }
    }

    // Check for error in response
    if (video_url.empty()) {
        std::string status = json_string_value(sse, "status");
        if (!status.empty()) {
            snprintf(res.error, sizeof(res.error), "Cosmos error: %.400s", status.c_str());
        } else {
            snprintf(res.error, sizeof(res.error), "No video path in response: %.400s", sse.c_str());
        }
        return res;
    }

    std::cout << "[cosmos] Result video: " << video_url << "\n";

    // ── 7. Download result video ────────────────────────────────────────────
    // The path may be a Gradio file URL like /gradio_api/file=<path>
    std::string dl_path = "/gradio_api/file=" + video_url;
    std::string video_data = http_get(host, port, dl_path.c_str(), https, 60000);
    if (video_data.empty()) {
        // Try direct path
        video_data = http_get(host, port, video_url.c_str(), https, 60000);
    }

    std::cout << "[cosmos] Downloaded: " << video_data.size() << " bytes\n";

    if (video_data.empty()) {
        snprintf(res.error, sizeof(res.error), "Failed to download result video");
        return res;
    }

    // ── 8. Extract first frame from result video ────────────────────────────
    // Try as image first
    int img_w = 0, img_h = 0, img_ch = 0;
    uint8_t* img = stbi_load_from_memory(
        reinterpret_cast<const uint8_t*>(video_data.data()),
        (int)video_data.size(), &img_w, &img_h, &img_ch, 4);
    if (img) {
        res.success = true;
        res.pixels  = new uint8_t[img_w * img_h * 4];
        memcpy(res.pixels, img, img_w * img_h * 4);
        res.width   = img_w;
        res.height  = img_h;
        stbi_image_free(img);
        std::cout << "[cosmos] Result image: " << img_w << "x" << img_h << "\n";
        return res;
    }

    // Try to find JPEG frame inside MP4 mdat
    const uint8_t* vd = reinterpret_cast<const uint8_t*>(video_data.data());
    for (size_t i = 0; i + 4 < video_data.size(); ++i) {
        if (vd[i] == 0xFF && vd[i+1] == 0xD8 && vd[i+2] == 0xFF) {
            img = stbi_load_from_memory(vd + i, (int)(video_data.size() - i),
                                         &img_w, &img_h, &img_ch, 4);
            if (img) {
                res.success = true;
                res.pixels  = new uint8_t[img_w * img_h * 4];
                memcpy(res.pixels, img, img_w * img_h * 4);
                res.width   = img_w;
                res.height  = img_h;
                stbi_image_free(img);
                std::cout << "[cosmos] Extracted frame: " << img_w << "x" << img_h << "\n";
                return res;
            }
        }
    }

    snprintf(res.error, sizeof(res.error),
             "Could not decode result video (%zu bytes)", video_data.size());
    return res;
}
