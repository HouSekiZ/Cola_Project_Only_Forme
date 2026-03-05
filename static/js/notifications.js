/**
 * notifications.js
 * จัดการ UI สำหรับ:
 *   - Repositioning reminder (progress bar + alert)
 *   - Meal time notifications (badge + panel)
 *
 * Dependencies: utils.js (fetchJSON, formatDuration)
 */

// ── Config ─────────────────────────────────────────────────────────────────
const NOTIFICATION_POLL_MS = 2000;    // poll pending notifications ทุก 2s
const POSITION_POLL_MS = 5000;    // poll body position ทุก 5s
const MEAL_POLL_MS = 30000;   // poll meal status ทุก 30s

// ── State ──────────────────────────────────────────────────────────────────
let _notificationPollTimer = null;
let _positionPollTimer = null;
let _mealPollTimer = null;
let _repositionSound = null;
let _mealSound = null;

// ── Init ───────────────────────────────────────────────────────────────────
function initNotifications() {
  _repositionSound = new Audio("/static/audio/reposition_notification.mp3");
  _mealSound = new Audio("/static/audio/meal_notification.mp3");

  _notificationPollTimer = setInterval(pollPendingNotifications, NOTIFICATION_POLL_MS);
  _positionPollTimer = setInterval(pollBodyPosition, POSITION_POLL_MS);
  _mealPollTimer = setInterval(pollMealStatus, MEAL_POLL_MS);

  // โหลดครั้งแรกทันที
  pollBodyPosition();
  pollMealStatus();
  populateMealTimeInputs(); // ดึงเวลาอาหารจาก server มาใส่ input
  loadNotificationHistory(); // โหลดประวัติจาก DB

  // ปุ่ม reset reposition timer (ID ตรงกับ index.html)
  const btn = document.getElementById("btnRepositionDone");
  if (btn) btn.addEventListener("click", onRepositionDone);
}

function destroyNotifications() {
  clearInterval(_notificationPollTimer);
  clearInterval(_positionPollTimer);
  clearInterval(_mealPollTimer);
}

// ── Polling ────────────────────────────────────────────────────────────────

async function pollPendingNotifications() {
  try {
    const notifications = await fetchJSON("/api/notifications/pending");
    notifications.forEach(handleNotification);
  } catch (e) {
    // silent fail — ไม่ spam console
  }
}

async function pollBodyPosition() {
  try {
    const status = await fetchJSON("/api/body_position");
    renderPositionWidget(status);
  } catch (e) { /* silent */ }
}

async function pollMealStatus() {
  try {
    const meals = await fetchJSON("/api/meal_times");
    renderMealPanel(meals);
  } catch (e) { /* silent */ }
}

// ── Notification Handler ───────────────────────────────────────────────────

function handleNotification(notif) {
  if (notif.type === "reposition") {
    showRepositionAlert(notif);
    playSound(_repositionSound);
  } else if (notif.type === "meal") {
    showMealAlert(notif);
    playSound(_mealSound);
  }
  addToNotificationLog(notif);
}

// ── Reposition UI ──────────────────────────────────────────────────────────

function renderPositionWidget(status) {
  // ── ป้ายท่านอนปัจจุบัน ──
  const posLabel = document.getElementById("currentPositionLabel");
  const posIcon = document.getElementById("positionIcon");

  const labels = {
    SUPINE: "หงาย",
    LEFT_SIDE: "ตะแคงซ้าย",
    RIGHT_SIDE: "ตะแคงขวา",
    UNKNOWN: "กำลังตรวจจับ...",
  };
  const icons = {
    SUPINE: "🛌",
    LEFT_SIDE: "◀️",
    RIGHT_SIDE: "▶️",
    UNKNOWN: "🔍",
  };

  if (posLabel) posLabel.textContent = labels[status.current_position] ?? status.current_position;
  if (posIcon) posIcon.textContent = icons[status.current_position] ?? "🔍";

  // ── Duration ──
  const durEl = document.getElementById("positionDuration");
  if (durEl) {
    durEl.textContent = status.current_position !== "UNKNOWN"
      ? `อยู่นาน: ${formatDuration(status.current_duration ?? 0)}`
      : "--";
  }

  // ── Progress bar เวลาก่อนพลิกตัว ──
  const progressBar = document.getElementById("repositionProgress");
  const timeLabel = document.getElementById("repositionTimeLabel");
  const interval = status.reposition_interval ?? 7200;
  const remaining = status.time_until_reposition ?? 0;
  const elapsed = interval - remaining;
  const pct = Math.min(100, Math.round((elapsed / interval) * 100));

  if (progressBar) {
    progressBar.style.width = `${pct}%`;
    progressBar.className = [
      "progress-fill",
      pct >= 90 ? "danger" : pct >= 70 ? "warning" : "normal",
    ].join(" ");
  }

  if (timeLabel) {
    timeLabel.textContent = status.reposition_due
      ? "⚠️ ถึงเวลาพลิกตัวแล้ว!"
      : `พลิกตัวในอีก ${formatDuration(remaining)}`;
  }
}

function showRepositionAlert(notif) {
  const el = document.getElementById("repositionAlert");
  if (!el) return;
  const msgEl = el.querySelector(".alert-message");
  if (msgEl) msgEl.textContent = notif.message;
  el.classList.remove("hidden");
}

async function onRepositionDone() {
  try {
    await fetchJSON("/api/reset_position_timer", { method: "POST" });
    const el = document.getElementById("repositionAlert");
    if (el) el.classList.add("hidden");
    await pollBodyPosition(); // refresh ทันที
  } catch (e) {
    console.error("Failed to reset reposition timer:", e);
  }
}

// ── Meal UI ────────────────────────────────────────────────────────────────

function renderMealPanel(meals) {
  const container = document.getElementById("mealStatusList");
  if (!container) return;

  container.innerHTML = "";
  meals.forEach((meal) => {
    const item = document.createElement("div");
    item.className = `meal-item ${meal.eaten ? "eaten" : "pending"}`;
    item.dataset.meal = meal.name;

    const mealNames = { breakfast: "เช้า", lunch: "เที่ยง", dinner: "เย็น" };
    item.innerHTML = `
      <span class="meal-icon">${meal.eaten ? "✅" : "🍽️"}</span>
      <span class="meal-name">อาหาร${mealNames[meal.name] ?? meal.name}</span>
      <span class="meal-time">${meal.scheduled_time}</span>
      ${!meal.eaten
        ? `<button class="btn-meal-eaten" onclick="markMealEaten('${meal.name}')">ทานแล้ว</button>`
        : '<span class="meal-eaten-badge">ทานแล้ว ✓</span>'
      }
    `;
    container.appendChild(item);
  });
}

function showMealAlert(notif) {
  // แสดง toast notification
  showToast(notif.title, notif.message, "meal");
}

async function markMealEaten(mealName) {
  try {
    await fetchJSON(`/api/meal_eaten/${mealName}`, { method: "POST" });
    await pollMealStatus(); // refresh panel ทันที
  } catch (e) {
    console.error("Failed to mark meal eaten:", e);
  }
}

// ── Toast ──────────────────────────────────────────────────────────────────

function showToast(title, message, type = "info") {
  const container = document.getElementById("toastContainer") ?? createToastContainer();
  const toast = document.createElement("div");
  toast.className = `toast toast-${type}`;
  toast.innerHTML = `
    <strong>${escapeHtml(title)}</strong>
    <p>${escapeHtml(message)}</p>
  `;
  container.appendChild(toast);
  // auto-remove หลัง 6 วินาที
  setTimeout(() => toast.remove(), 6000);
}

function createToastContainer() {
  const el = document.createElement("div");
  el.id = "toast-container";
  document.body.appendChild(el);
  return el;
}

// ── Notification Log ───────────────────────────────────────────────────────

function addToNotificationLog(notif) {
  const log = document.getElementById("notificationLog");
  if (!log) return;

  // ลบข้อความ placeholder ถ้ามี
  const empty = log.querySelector(".log-empty");
  if (empty) empty.remove();

  const entry = document.createElement("div");
  entry.className = `log-entry log-${notif.type}`;
  const time = new Date(notif.timestamp * 1000).toLocaleTimeString("th-TH");
  entry.innerHTML = `
    <span class="log-time">${time}</span>
    <span class="log-title">${escapeHtml(notif.title)}</span>
  `;
  log.prepend(entry);

  // เก็บไม่เกิน 20 รายการ
  const entries = log.querySelectorAll(".log-entry");
  if (entries.length > 20) entries[entries.length - 1].remove();
}


// ── Load DB History on Page Load ───────────────────────────────────────────

async function loadNotificationHistory() {
  const log = document.getElementById("notificationLog");
  if (!log) return;

  try {
    // ดึงทั้ง notification history และ alarm events จาก DB
    const [notifications, alarms] = await Promise.all([
      fetchJSON("/api/db/notification_history?limit=20").catch(() => []),
      fetchJSON("/api/db/alarm_history?limit=20").catch(() => []),
    ]);

    // รวมและเรียงตามเวลา (ใหม่สุดก่อน)
    const combined = [
      ...notifications.map(n => ({
        type: n.type || "reposition",
        title: n.title,
        timestamp: new Date(n.created_at).getTime() / 1000,
        source: "notification",
      })),
      ...alarms.map(a => ({
        type: "alarm",
        title: `🚨 ${_alarmLabel(a.alarm_type)}`,
        timestamp: new Date(a.triggered_at).getTime() / 1000,
        source: "alarm",
      }))
    ].sort((a, b) => b.timestamp - a.timestamp).slice(0, 30);

    if (combined.length === 0) return;

    // ลบ placeholder ถ้ามี
    const empty = log.querySelector(".log-empty");
    if (empty) empty.remove();

    combined.forEach(item => {
      const entry = document.createElement("div");
      entry.className = `log-entry log-${item.type}`;
      const d = new Date(item.timestamp * 1000);
      const timeStr = d.toLocaleDateString("th-TH", { day: "2-digit", month: "2-digit" })
        + " " + d.toLocaleTimeString("th-TH", { hour: "2-digit", minute: "2-digit" });
      entry.innerHTML = `
        <span class="log-time">${timeStr}</span>
        <span class="log-title">${escapeHtml(item.title)}</span>
      `;
      log.appendChild(entry);
    });

    // อัปเดต badge count
    const badge = document.querySelector(".notif-count");
    if (badge) badge.textContent = combined.length;

  } catch (e) {
    // DB ไม่พร้อม — ใช้ in-memory แทน (ไม่แสดง error)
    console.debug("loadNotificationHistory: DB not available", e);
  }
}

function _alarmLabel(type) {
  return { eye_blink: "Eye Blink", hand_gesture: "Hand Gesture", body_position: "Body Position" }[type] || type;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function playSound(audio) {
  if (!audio) return;
  audio.currentTime = 0;
  audio.play().catch(() => { /* user hasn't interacted yet — browser policy */ });
}

function formatDuration(seconds) {
  if (seconds <= 0) return "0 นาที";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  if (h > 0) return `${h} ชม. ${m} นาที`;
  return `${m} นาที`;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ── Meal Time Settings ─────────────────────────────────────────────────────

async function saveMealTimes() {
  const breakfast = document.getElementById("mealTimeBreakfast")?.value;
  const lunch = document.getElementById("mealTimeLunch")?.value;
  const dinner = document.getElementById("mealTimeDinner")?.value;
  const statusEl = document.getElementById("mealSaveStatus");

  if (!breakfast || !lunch || !dinner) return;

  try {
    await fetchJSON("/api/meal_times", {
      method: "POST",
      body: JSON.stringify({ breakfast, lunch, dinner }),
    });
    if (statusEl) {
      statusEl.textContent = "✓ บันทึกแล้ว";
      statusEl.className = "meal-save-status success";
      setTimeout(() => { statusEl.textContent = ""; }, 3000);
    }
    await pollMealStatus(); // refresh ทันที
  } catch (e) {
    if (statusEl) {
      statusEl.textContent = "⚠️ บันทึกไม่สำเร็จ";
      statusEl.className = "meal-save-status error";
    }
    console.error("Failed to save meal times:", e);
  }
}

async function populateMealTimeInputs() {
  try {
    const meals = await fetchJSON("/api/meal_times");
    if (!Array.isArray(meals)) return;
    meals.forEach((meal) => {
      const idMap = {
        breakfast: "mealTimeBreakfast",
        lunch: "mealTimeLunch",
        dinner: "mealTimeDinner",
      };
      const el = document.getElementById(idMap[meal.name]);
      if (el && meal.scheduled_time) el.value = meal.scheduled_time;
    });
  } catch (_) { /* API ยังไม่พร้อม — ใช้ค่า default */ }
}

// ── Export ─────────────────────────────────────────────────────────────────
window.NotificationsModule = {
  init: initNotifications,
  destroy: destroyNotifications,
  markMealEaten,
  refreshPosition: pollBodyPosition,
};
// expose saveMealTimes globally (used by inline onclick)
window.saveMealTimes = saveMealTimes;
window.markMealEaten = markMealEaten;
