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
const POSITION_POLL_MS     = 5000;    // poll body position ทุก 5s
const MEAL_POLL_MS         = 30000;   // poll meal status ทุก 30s

// ── State ──────────────────────────────────────────────────────────────────
let _notificationPollTimer = null;
let _positionPollTimer     = null;
let _mealPollTimer         = null;
let _repositionSound       = null;
let _mealSound             = null;

// ── Init ───────────────────────────────────────────────────────────────────
function initNotifications() {
  _repositionSound = new Audio("/static/audio/reposition_notification.mp3");
  _mealSound       = new Audio("/static/audio/meal_notification.mp3");

  _notificationPollTimer = setInterval(pollPendingNotifications, NOTIFICATION_POLL_MS);
  _positionPollTimer     = setInterval(pollBodyPosition, POSITION_POLL_MS);
  _mealPollTimer         = setInterval(pollMealStatus, MEAL_POLL_MS);

  // โหลดครั้งแรกทันที
  pollBodyPosition();
  pollMealStatus();

  // ปุ่ม reset reposition timer
  const btn = document.getElementById("btn-reposition-done");
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
  const posLabel = document.getElementById("current-position-label");
  const posIcon  = document.getElementById("current-position-icon");
  if (posLabel) {
    const labels = {
      SUPINE:     "หงาย",
      LEFT_SIDE:  "ตะแคงซ้าย",
      RIGHT_SIDE: "ตะแคงขวา",
      UNKNOWN:    "ไม่ทราบ",
    };
    posLabel.textContent = labels[status.current_position] ?? status.current_position;
  }

  // ── Progress bar เวลาก่อนพลิกตัว ──
  const progressBar = document.getElementById("reposition-progress");
  const timeLabel   = document.getElementById("reposition-time-label");
  const interval    = status.reposition_interval ?? 7200;
  const remaining   = status.time_until_reposition ?? 0;
  const elapsed     = interval - remaining;
  const pct         = Math.min(100, Math.round((elapsed / interval) * 100));

  if (progressBar) {
    progressBar.style.width = `${pct}%`;
    progressBar.className = [
      "reposition-progress-fill",
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
  const el = document.getElementById("reposition-alert");
  if (!el) return;
  el.querySelector(".alert-message").textContent = notif.message;
  el.classList.remove("hidden");
}

async function onRepositionDone() {
  try {
    await fetchJSON("/api/reset_position_timer", { method: "POST" });
    const el = document.getElementById("reposition-alert");
    if (el) el.classList.add("hidden");
    await pollBodyPosition(); // refresh ทันที
  } catch (e) {
    console.error("Failed to reset reposition timer:", e);
  }
}

// ── Meal UI ────────────────────────────────────────────────────────────────

function renderMealPanel(meals) {
  const container = document.getElementById("meal-status-list");
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
        ? `<button class="btn-meal-eaten" onclick="markMealEaten('${meal.name}')">
             ทานแล้ว
           </button>`
        : '<span class="meal-eaten-badge">ทานแล้ว</span>'
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
  const container = document.getElementById("toast-container") ?? createToastContainer();
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
  const log = document.getElementById("notification-log");
  if (!log) return;

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

// ── Export ─────────────────────────────────────────────────────────────────
// ถ้าใช้ ES modules:
// export { initNotifications, destroyNotifications, markMealEaten };

// ถ้าใช้ script tag ปกติ:
window.NotificationsModule = {
  init:          initNotifications,
  destroy:       destroyNotifications,
  markMealEaten,
};
