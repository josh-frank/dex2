/**
 * display.h
 *
 * All OLED output for e-meter.ino.
 * Cycle display modes with the button on the XIAO Expansion Board (D1).
 *
 * Modes:
 *   0  DISP_POLYGRAM  — full-screen 30-second rolling waveform
 *   1  DISP_NUMBERS   — µS + clock + status icons + min/max + waveform strip
 *
 * Public API (call from e-meter.ino):
 *   display_splash()               — boot screen, call after u8g2.begin()
 *   display_splash_status(msg)     — update the status line on the boot screen
 *   display_setup()                — wires up button ISR, call after display_splash()
 *   display_render(uS, dt, unix_t) — render current mode, call every loop()
 *
 * Externs required in e-meter.ino (must NOT be static):
 *   PolyPoint g_poly[POLY_MAX];
 *   int       g_poly_head;
 *   int       g_poly_len;
 */

#pragma once
#include <Arduino.h>
#include <U8g2lib.h>
#ifdef USE_WIFI
  #include <WiFi.h>
#endif

// ── Constants ─────────────────────────────────────────────────────────────────
#define OLED_W    128
#define OLED_H    64
#define POLY_MAX  600
#define WINDOW_MS 30000UL

// ── PolyPoint ─────────────────────────────────────────────────────────────────
struct PolyPoint { uint32_t t_ms; float uS; };

// ── Externs — defined in e-meter.ino ─────────────────────────────────────────
extern U8G2_SSD1306_128X64_NONAME_F_HW_I2C u8g2;
extern PolyPoint g_poly[];
extern int       g_poly_head;
extern int       g_poly_len;

// Transport state externs — conditionally compiled to match e-meter.ino
#ifdef USE_SD
  extern bool g_recording;
#endif
#ifdef USE_BLE
  extern bool g_bleConnected;
#endif

// DateTime forward declaration — full definition comes from PCF8563.h
struct DateTime;

// ── Button ────────────────────────────────────────────────────────────────────
static const uint8_t  BTN_PIN      = D1;
static const uint32_t BTN_DEBOUNCE = 250;   // ms

// ── Display mode ──────────────────────────────────────────────────────────────
#define DISP_POLYGRAM    0
#define DISP_NUMBERS     1
// #define DISP_PLACEHOLDER 2   // ← uncomment + set DISP_MODE_MAX 3 to add a mode
#define DISP_MODE_MAX    2

volatile uint8_t  g_display_mode = DISP_POLYGRAM;
volatile uint32_t g_btn_last_ms  = 0;

// ── ISR ───────────────────────────────────────────────────────────────────────
void IRAM_ATTR btn_isr() {
  uint32_t now = millis();
  if (now - g_btn_last_ms > BTN_DEBOUNCE) {
    g_display_mode = (g_display_mode + 1) % DISP_MODE_MAX;
    g_btn_last_ms  = now;
  }
}

// ── display_splash — boot screen ──────────────────────────────────────────────
void display_splash() {
  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_5x7_tr);
  u8g2.drawStr(0, 8, "loading...");
  u8g2.drawHLine(0, 12, OLED_W);
  u8g2.sendBuffer();
}

void display_splash_status(const char* msg) {
  // Erase just the status area and redraw — header stays put
  u8g2.setDrawColor(0);
  u8g2.drawBox(0, 13, OLED_W, OLED_H - 13);
  u8g2.setDrawColor(1);
  u8g2.setFont(u8g2_font_5x7_tr);
  u8g2.drawStr(0, 22, msg);
  u8g2.sendBuffer();
}

// ── display_setup — call once after display_splash() ─────────────────────────
void display_setup() {
  pinMode(BTN_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(BTN_PIN), btn_isr, FALLING);
}

static void unix_to_b32(char* buf, size_t len, uint32_t unix_t) {
  const char* digits = "0123456789ABCDEFGHIJKLMNOPQRSTUV";
  char tmp[14];
  int  i = 0;
  if (unix_t == 0) { buf[0] = '0'; buf[1] = '\0'; return; }
  while (unix_t > 0) {
    tmp[i++] = digits[unix_t % 32];
    unix_t  /= 32;
  }
  // reverse
  int j = 0;
  while (i > 0) buf[j++] = tmp[--i];
  buf[j] = '\0';
}

// ─────────────────────────────────────────────────────────────────────────────
//  Waveform helpers (shared by POLYGRAM and NUMBERS)
// ─────────────────────────────────────────────────────────────────────────────

// Min/max µS over the current window. Returns false if too few points.
static bool poly_bounds(float& lo, float& hi) {
  if (g_poly_len < 2) return false;
  lo = hi = g_poly[g_poly_head].uS;
  for (int i = 1; i < g_poly_len; i++) {
    float v = g_poly[(g_poly_head + i) % POLY_MAX].uS;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  return true;
}

// Draw the rolling waveform into the band y0 .. y0+h-1, auto-scaled to the
// window's min/max with 20 % padding (min 2 µS) so it never sits on the edges.
static void draw_poly(int y0, int h) {
  float lo, hi;
  if (!poly_bounds(lo, hi)) return;

  uint32_t now = millis();

  float pad   = max((hi - lo) * 0.2f, 2.0f);
  float vmin  = lo - pad;
  float range = (hi + pad) - (lo - pad);
  if (range < 1.0f) range = 1.0f;

  int prev_px = -1, prev_py = -1;
  for (int i = 0; i < g_poly_len; i++) {
    PolyPoint& p = g_poly[(g_poly_head + i) % POLY_MAX];

    float age_ratio = (float)(now - p.t_ms) / (float)WINDOW_MS;
    int px = constrain((int)(OLED_W * (1.0f - age_ratio)), 0, OLED_W - 1);

    float norm_y = 1.0f - (p.uS - vmin) / range;      // higher µS = higher on screen
    int py = constrain(y0 + (int)(norm_y * (h - 1)), y0, y0 + h - 1);

    if (prev_px >= 0) u8g2.drawLine(prev_px, prev_py, px, py);
    prev_px = px;
    prev_py = py;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mode 0 — POLYGRAM
//  Full-screen 30-second rolling waveform, no labels.
// ─────────────────────────────────────────────────────────────────────────────
static void render_polygram() {
  draw_poly(0, OLED_H);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mode 1 — NUMBERS
//  Layout (128x64):
//    y=0..13   µS reading (large font)
//    y=14..23  clock (left) + transport status icons (right)
//    y=24..32  window min / max µS
//    y=43      divider
//    y=44..63  waveform strip (20 px)
//
//  Status icons (5x7 font, right-aligned):
//    W / w  — WiFi connected / offline
//    B / b  — BLE connected / advertising
//    * / -  — SD recording / mounted idle
//
//  µ glyph note:
//    drawStr() treats strings as Latin-1, so the UTF-8 µ (0xC2 0xB5)
//    renders as two glyphs. Use drawUTF8() with a _tf font instead.
// ─────────────────────────────────────────────────────────────────────────────
static void render_numbers(float uS, const DateTime& dt, uint32_t unix_t) {
  // Row 1 — µS value
  u8g2.setFont(u8g2_font_7x14B_tf);
  char us_str[20];
  snprintf(us_str, sizeof(us_str), "%.2f uS", uS);
  u8g2.drawUTF8(0, 13, us_str);

  // Row 2 — clock left, status icons right
  u8g2.setFont(u8g2_font_5x7_tr);
  char time_str[24];   // HH:MM:SS + space + up to 7 b32 chars
  char b32_str[14];
  unix_to_b32(b32_str, sizeof(b32_str), unix_t);
  int display_hour = (dt.hour + TZ_OFFSET + 24) % 24;
  snprintf(time_str, sizeof(time_str), "%02d:%02d:%02d %s",
          display_hour, dt.minute, dt.second, b32_str);
  u8g2.drawStr(0, 23, time_str);

  char status[8] = "";
  int  scol = 0;
  #ifdef USE_SD
    status[scol++] = g_recording ? '*' : '-';
  #endif
  #ifdef USE_BLE
    status[scol++] = g_bleConnected ? 'B' : 'b';
  #endif
  #ifdef USE_WIFI
    status[scol++] = (WiFi.status() == WL_CONNECTED) ? 'W' : 'w';
  #endif
  status[scol] = '\0';
  for (int i = 0, j = scol - 1; i < j; i++, j--) {
    char tmp = status[i]; status[i] = status[j]; status[j] = tmp;
  }
  u8g2.drawStr(OLED_W - scol * 6, 23, status);

  // Row 3 — min / max over the 30 s window
  float lo, hi;
  if (poly_bounds(lo, hi)) {
    u8g2.setFont(u8g2_font_5x7_tf);
    char mm_str[32];
    snprintf(mm_str, sizeof(mm_str), "lo:%.2f hi:%.2f", lo, hi);
    u8g2.drawStr(0, 32, mm_str);
  }

  // Divider + waveform strip (y=44..63)
  u8g2.drawHLine(0, 43, OLED_W);
  draw_poly(44, OLED_H - 44);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Mode 2 — PLACEHOLDER  (commented out)
//
//  To activate:
//    1. Uncomment #define DISP_PLACEHOLDER 2 above
//    2. Change DISP_MODE_MAX to 3
//    3. Uncomment render_placeholder() and the case in display_render() below
// ─────────────────────────────────────────────────────────────────────────────
// static void render_placeholder() {
//   // your new mode here
// }

// ─────────────────────────────────────────────────────────────────────────────
//  display_render — call every frame from loop()
// ─────────────────────────────────────────────────────────────────────────────
void display_render(float uS, const DateTime& dt, uint32_t unix_t) {
  u8g2.clearBuffer();

  switch (g_display_mode) {
    case DISP_POLYGRAM: render_polygram();                      break;
    case DISP_NUMBERS:  render_numbers(uS, dt, unix_t);         break;
  }

  u8g2.sendBuffer();
}
