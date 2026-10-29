// Numbered overlay and 3x3 mouse grid rendering for Reach.

const root = document.getElementById("overlay-root");

let activeBadges = [];
let activeGrid = null;

// Re-renders badges and grid on the screen.
function render() {
  root.innerHTML = "";

  if (activeBadges && activeBadges.length > 0) {
    for (const badge of activeBadges) {
      const el = document.createElement("div");
      el.className = "number-badge";
      el.textContent = badge.number;
      el.style.left = `${badge.rect.left}px`;
      el.style.top = `${badge.rect.top}px`;
      root.appendChild(el);
    }
  }

  if (activeGrid) {
    const container = document.createElement("div");
    container.className = "grid-container";
    container.style.left = `${activeGrid.bounds.left}px`;
    container.style.top = `${activeGrid.bounds.top}px`;
    container.style.width = `${activeGrid.bounds.right - activeGrid.bounds.left}px`;
    container.style.height = `${activeGrid.bounds.bottom - activeGrid.bounds.top}px`;

    const levelBadge = document.createElement("div");
    levelBadge.className = "grid-level-indicator";
    levelBadge.textContent = `GRID LEVEL ${activeGrid.level}`;
    container.appendChild(levelBadge);

    for (const item of activeGrid.cells) {
      const cellEl = document.createElement("div");
      cellEl.className = "grid-cell";
      const cellLeft = item.rect.left - activeGrid.bounds.left;
      const cellTop = item.rect.top - activeGrid.bounds.top;
      const cellWidth = item.rect.right - item.rect.left;
      const cellHeight = item.rect.bottom - item.rect.top;

      cellEl.style.left = `${cellLeft}px`;
      cellEl.style.top = `${cellTop}px`;
      cellEl.style.width = `${cellWidth}px`;
      cellEl.style.height = `${cellHeight}px`;

      const numEl = document.createElement("div");
      numEl.className = "grid-number";
      numEl.textContent = item.cell;
      cellEl.appendChild(numEl);

      container.appendChild(cellEl);
    }

    root.appendChild(container);
  }
}

// Dispatches overlay events received from Tauri.
function handleOverlayEvent(payload) {
  if (!payload || !payload.type) return;

  switch (payload.type) {
    case "show_numbers":
      activeBadges = payload.badges || [];
      render();
      break;
    case "hide_numbers":
      activeBadges = [];
      render();
      break;
    case "show_grid":
      activeGrid = {
        bounds: payload.bounds,
        level: payload.level,
        cells: payload.cells || [],
      };
      render();
      break;
    case "hide_grid":
      activeGrid = null;
      render();
      break;
    case "clear":
      activeBadges = [];
      activeGrid = null;
      render();
      break;
    default:
      break;
  }
}

// Subscribes to the reach-overlay event from Tauri.
async function init() {
  try {
    const { listen } = window.__TAURI__.event;
    await listen("reach-overlay", (event) => handleOverlayEvent(event.payload));
  } catch (err) {
    // Front end fails silently if run outside Tauri
  }
}

init();
