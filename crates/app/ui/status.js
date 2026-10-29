// Status window rendering for the Reach voice loop.
//
// The runtime sends one of a fixed set of events. Error events carry a kind,
// never free-form text, so this file maps them to safe captions.
//
// Every event arrives in an envelope with a monotonic sequence number. The
// listener is registered before the snapshot is requested, so no early event
// is missed; sequence numbers then discard duplicates and stale snapshot
// entries.

const card = document.getElementById("card");
const indicator = document.getElementById("indicator");
const state = document.getElementById("state");
const heard = document.getElementById("heard");
const result = document.getElementById("result");

const ERROR_MESSAGES = {
  model: "The language model is unavailable.",
  execution: "The action could not be completed.",
  io: "The voice loop lost its input.",
  internal: "The voice loop could not start.",
};

let currentState = "asleep";
let lastSeq = -1;
let pendingEnvelopes = [];
let ready = false;
let actionHoldTimer = null;

// Clears any pending timer that would transition Action to Asleep.
function clearActionHold() {
  if (actionHoldTimer !== null) {
    clearTimeout(actionHoldTimer);
    actionHoldTimer = null;
  }
}

// Applies the colour shared by the card border and the indicator.
function setTone(tone) {
  card.dataset.tone = tone;
  indicator.dataset.tone = tone;
}

// Sets the large state word, its colour and the state key used by later
// events to decide whether the last result should be kept.
function setState(label, tone, nextState) {
  currentState = nextState;
  state.textContent = label;
  setTone(tone);
}

// Shows the last transcript the loop accepted.
function setHeard(text) {
  heard.textContent = `Heard: “${text}”`;
}

// Shows the last result or instruction.
function setResult(text) {
  result.textContent = text;
}

// Renders one status event from the shared voice loop.
function applyStatus(status) {
  if (!status || typeof status.type !== "string") {
    return;
  }

  switch (status.type) {
    case "listening":
      clearActionHold();
      setState("LISTENING", "listening", "listening");
      setResult("Listening for your command.");
      break;
    case "asleep": {
      const previousState = currentState;
      if (previousState === "action") {
        // Keep the Action result visible for 2 seconds before returning to Asleep.
        clearActionHold();
        actionHoldTimer = setTimeout(() => {
          actionHoldTimer = null;
          if (currentState === "action") {
            setState("ASLEEP", "asleep", "asleep");
          }
        }, 2000);
        break;
      }
      clearActionHold();
      setState("ASLEEP", "asleep", "asleep");
      // Keep the last action or clarification answer visible, but do not leave
      // a stale confirmation prompt or "listening" instruction on screen.
      if (previousState === "confirming") {
        setResult("Cancelled.");
      } else if (previousState === "listening") {
        setResult("Say “Reach” to start.");
      }
      break;
    }
    case "heard":
      clearActionHold();
      setHeard(status.text || "");
      break;
    case "action":
      clearActionHold();
      setState("DONE", "listening", "action");
      setResult(status.label || "Action completed.");
      break;
    case "confirming":
      clearActionHold();
      setState("CONFIRM?", "confirming", "confirming");
      setResult(`Say yes or no: ${status.label || "action"}`);
      break;
    case "not_understood":
      clearActionHold();
      setState("NOT UNDERSTOOD", "confirming", "not_understood");
      setResult("Please say it another way.");
      break;
    case "stopped":
      clearActionHold();
      // A startup model or executor failure is followed by `Stopped`. Keep
      // the error visible instead of masking it with a generic stop caption.
      if (currentState === "error") {
        setResult("The voice loop stopped after an error.");
      } else {
        setState("STOPPED", "stopped", "stopped");
        setResult("The voice loop has stopped.");
      }
      break;
    case "error":
      clearActionHold();
      setState("ERROR", "error", "error");
      setResult(ERROR_MESSAGES[status.kind] || "Something went wrong.");
      break;
    default:
      break;
  }
}

// Applies an envelope once, in sequence order.
function applyEnvelope(envelope) {
  if (!envelope || typeof envelope.seq !== "number") {
    return;
  }
  if (envelope.seq <= lastSeq) {
    return;
  }
  lastSeq = envelope.seq;
  applyStatus(envelope.status);
}

// Buffers live events until the snapshot has been replayed.
function receiveEnvelope(envelope) {
  if (!ready) {
    pendingEnvelopes.push(envelope);
    return;
  }
  applyEnvelope(envelope);
}

// Registers the event listener, then fills in any event that was emitted
// before this script finished loading.
async function start() {
  try {
    const { listen } = window.__TAURI__.event;
    const { invoke } = window.__TAURI__.core;

    await listen("reach-status", (event) => receiveEnvelope(event.payload));
    const snapshot = (await invoke("current_status")) || [];
    ready = true;
    for (const envelope of snapshot) {
      applyEnvelope(envelope);
    }
    const buffered = pendingEnvelopes;
    pendingEnvelopes = [];
    for (const envelope of buffered) {
      applyEnvelope(envelope);
    }
  } catch {
    ready = true;
    setState("ERROR", "error", "error");
    setResult("The status channel could not start.");
  }
}

start();
