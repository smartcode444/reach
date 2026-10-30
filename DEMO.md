# Reach Demo Walkthrough

This document outlines the exact demo scenarios, spoken/typed utterances, and expected behavior supported by the Reach implementation.

Because the `audio` and `stt` capture crates are currently scaffolding (`[not built]`), the demo can be driven via the interactive CLI pipeline or the Tauri status HUD using typed input standing in for speech.

---

## 1. Starting the Demo

### Option A: Interactive CLI Pipeline
```powershell
# Run the pipeline with the local offline rule grammar:
cargo run -p cli -- pipeline

# Or run with the in-process Needle 3 model loaded:
cargo run -p cli -- pipeline --model models/needle3.cact
```

### Option B: Tauri Always-On-Top Status HUD
```powershell
cargo run -p reach-app
```
*Note: Type commands into the console window that launched the application; the transparent HUD window in the bottom-right corner will update its state, indicator, and captions without taking keyboard focus.*

---

## 2. Demo Scenarios & Exact Utterances

### Scenario 1: Waking the Assistant
The assistant ignores background speech until the wake word is heard.

| Step | Utterance | Expected Route | Status HUD Display | Result Description |
|---|---|---|---|---|
| 1.1 | `good morning` | Ignored | `ASLEEP` | Speech ignored; listening window remains closed. |
| 1.2 | `Reach` | WakeOnly | `LISTENING` | Listening window opens for 5 seconds. |

---

### Scenario 2: Fast-Path Rule Execution
Common navigation commands execute immediately on the fast-path without invoking the language model.

| Step | Utterance | Expected Route | Status HUD Display | Result Description |
|---|---|---|---|---|
| 2.1 | `Reach scroll down` | `Route::Run` | `DONE` (Action) | Active window scrolls down by 3 steps. |
| 2.2 | `Reach zoom in` | `Route::Run` | `DONE` (Action) | Sends unshifted `Ctrl+=` to zoom in active browser. |
| 2.3 | `Reach press enter` | `Route::Run` | `DONE` (Action) | Injects Enter keypress and releases modifiers. |

---

### Scenario 3: Screen Element Interaction (UI Automation)
Clicking and focusing elements on the foreground application.

| Step | Utterance | Expected Route | Status HUD Display | Result Description |
|---|---|---|---|---|
| 3.1 | `Reach focus field First Name` | `Route::Run` | `DONE` (Action) | Locates editable field matching "First Name" and moves focus. |
| 3.2 | `Reach type John Doe` | `Route::Run` | `DONE` (Action) | Injects keyboard text into the focused element. |
| 3.3 | `Reach click Search` | `Route::Run` | `DONE` (Action) | Finds clickable element with accessible label "Search" and clicks its center. |

---

### Scenario 4: Safety & Confirmation for Risky Actions
Actions flagged as potentially destructive or irreversible require explicit confirmation.

| Step | Utterance | Expected Route | Status HUD Display | Result Description |
|---|---|---|---|---|
| 4.1 | `Reach click Submit` | `Route::Confirm` | `CONFIRM?` | System pauses: *"Say yes or no: click element 'Submit'"*. |
| 4.2 | `yes` | Confirmed | `DONE` (Action) | Action confirmed; clicks "Submit" button. |
| 4.3 | `Reach click Delete` | `Route::Confirm` | `CONFIRM?` | System pauses: *"Say yes or no: click element 'Delete'"*. |
| 4.4 | `no` | Denied | `ASLEEP` | Action cancelled; no input injected. |

---

### Scenario 5: Application & Web Launching
Executes strictly against the approved executable allow-list and validated URL schemes.

| Step | Utterance | Expected Route | Status HUD Display | Result Description |
|---|---|---|---|---|
| 5.1 | `Reach open notepad` | `Route::Run` | `DONE` (Action) | Launches `notepad.exe`. |
| 5.2 | `Reach open chrome` | `Route::Run` | `DONE` (Action) | Launches Google Chrome with `--force-renderer-accessibility`. |
| 5.3 | `Reach open https://example.com` | `Route::Run` | `DONE` (Action) | Opens validated URL in Chrome with accessibility enabled. |

---

### Scenario 6: Disambiguation & Clarification
When an utterance cannot be matched or is ambiguous, Reach clarifies rather than guessing.

| Step | Utterance | Expected Route | Status HUD Display | Result Description |
|---|---|---|---|---|
| 6.1 | `Reach do something cool` | `Route::Clarify` | `NOT UNDERSTOOD` | Prompts: *"I didn't catch that, could you say it another way?"*. |

---

### Scenario 7: Emergency Stop & Resumption
Halts all pipeline execution and resets physical/virtual input state.

1. **Voice Emergency Stop (Accessible Method):**
   - Utterance: `Reach stop`
   - Outcome: Immediately transitions Reach to `Mode::Sleep`, drops pending confirmations, and neutralizes simulated keys.
2. **Global Physical Hotkey (Operator / Caregiver Method):**
   - Press: `Ctrl + Alt + Q`
   - Outcome: Sets atomic STOP flag on dedicated message loop thread, immediately releases all modifier keys and mouse buttons via `release_all_keys()`, and aborts executor actions.
3. **Resumption:**
   - Press: `Ctrl + Alt + R`
   - Outcome: Clears emergency stop flag and returns system to normal standby mode.

---

### Scenario 8: Numbered Overlay & 3x3 Mouse Grid
Visual fallbacks when elements lack accessible names or target fine-grained canvas positions.

| Step | Utterance | Expected Route | Status HUD Display | Result Description |
|---|---|---|---|---|
| 8.1 | `Reach show numbers` | `Route::Run` | `DONE` (Action) | Collects interactive elements and draws yellow badges with stable 1-based numbers. |
| 8.2 | `Reach click 2` | `Route::Run` | `DONE` (Action) | Clicks element #2 from the active snapshot and clears the overlay. |
| 8.3 | `Reach hide numbers` | `Route::Run` | `DONE` (Action) | Clears the snapshot and hides numbered badges. |
| 8.4 | `Reach show grid` | `Route::Run` | `DONE` (Action) | Displays a 3x3 mouse grid numbered 1 to 9 over the foreground window rect. |
| 8.5 | `Reach choose 5` | `Route::Run` | `DONE` (Action) | Zooms grid into cell 5 (level 2). |
| 8.6 | `Reach choose 1` | `Route::Run` | `DONE` (Action) | Zooms grid into cell 1 (level 3). |
| 8.7 | `Reach choose 9` | `Route::Run` | `DONE` (Action) | On final level, clicks cell centre at calculated coordinates and hides grid. |

---

## 3. Demo Limitations
- Live microphone audio capture and Whisper STT are **[not built]**; typed console input is used in this demonstration.
