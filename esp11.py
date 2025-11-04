# ESP Red vs Blue Pill Task (Final v1)
# Yes/No SDT version with feedback toggle, ROC + d', c, and binomial test

from psychopy import visual, event, core, gui
import random, csv, os
from datetime import datetime

# ---------- Imports ----------
try:
    from scipy.stats import norm
    try:
        from scipy.stats import binom_test  # older SciPy
    except ImportError:
        from scipy.stats import binomtest as binom_test  # newer SciPy
    SCIPY_PRESENT = True
except Exception:
    SCIPY_PRESENT = False

# ---------- Participant Info ----------
info = {"Participant ID": "P01", "Session": "1", "Trials": 100, "Show Feedback?": True}
dlg = gui.DlgFromDict(info, title="ESP Red vs Blue Pill Task (SDT)")
if not dlg.OK:
    core.quit()

participant = info["Participant ID"]
session = info["Session"]
n_trials = int(info["Trials"])
show_feedback = bool(info["Show Feedback?"])

# ---------- Window & Stimuli ----------
win = visual.Window(fullscr=True, color="black", units="norm")
msg = visual.TextStim(win, text="", color="white", height=0.08, wrapWidth=1.6)
fix = visual.TextStim(win, text="+", color="white", height=0.15)

# ---------- Helper ----------
def show_text_block(text, duration=None, wait_for_keys=True):
    msg.text = text
    msg.pos = (0, 0)
    msg.draw()
    win.flip()
    if duration:
        core.wait(duration)
    elif wait_for_keys:
        event.waitKeys()

# ---------- Output Files ----------
outdir = "esp_redblue_data"
os.makedirs(outdir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trialfile = os.path.join(outdir, f"{participant}_sess{session}_{timestamp}_trials.csv")
summaryfile = os.path.join(outdir, f"{participant}_sess{session}_{timestamp}_summary.csv")
rocfile = os.path.join(outdir, f"{participant}_sess{session}_{timestamp}_ROC.csv")

with open(trialfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["participant", "session", "trial", "signal_present", "response",
                     "correct", "confidence", "rt_seconds", "timestamp"])

# ---------- Intro ----------
show_text_block(
    "Welcome to the ESP Red vs Blue Pill Task!\n\n"
    "In each trial, imagine a pill hidden inside the box.\n"
    "It can be *Red* (signal) or *Blue* (no signal).\n\n"
    "Decide whether you *sense* a Red pill (press Y) or not (press N).\n"
    "Then rate your confidence (1–5).\n\nPress SPACE to start."
)
event.waitKeys(keyList=["space"])

clock = core.Clock()
rows = []

# ---------- Trial Loop ----------
for trial in range(1, n_trials + 1):
    signal_present = random.choice([True, False])  # True = red pill
    fix.draw()
    win.flip()
    core.wait(0.5)

    msg.text = f"Trial {trial}/{n_trials}\n\nDo you sense a *Red pill* inside the box?\n\nPress Y (Yes) or N (No)."
    msg.pos = (0, 0.2)
    msg.draw()
    box = visual.Rect(win, width=0.4, height=0.4, pos=(0, -0.2), fillColor='white', lineColor='white')
    box.draw()
    win.flip()

    clock.reset()
    keys = event.waitKeys(keyList=['y', 'n', 'escape'])
    if 'escape' in keys:
        break

    response = 'yes' if 'y' in keys else 'no'
    rt = clock.getTime()
    correct = int((signal_present and response == 'yes') or (not signal_present and response == 'no'))

    # Confidence rating
    msg.text = "Rate your confidence:\n1 (least sure) → 5 (most sure)"
    msg.pos = (0, 0)
    msg.draw()
    win.flip()
    conf_key = event.waitKeys(keyList=['1', '2', '3', '4', '5', 'escape'])[0]
    if conf_key == 'escape':
        break
    confidence = int(conf_key)

    # Feedback
    if show_feedback:
        fb_text = "✅ Correct!" if correct else f"❌ Wrong. It was {'Red' if signal_present else 'Blue'}."
        msg.text = fb_text
        msg.pos = (0, 0)
        msg.draw()
        win.flip()
        core.wait(1.0)

    rows.append({
        'trial': trial,
        'signal_present': signal_present,
        'response': response,
        'correct': correct,
        'confidence': confidence,
        'rt_seconds': round(rt, 3),
        'timestamp': datetime.now().isoformat()
    })

    with open(trialfile, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([participant, session, trial, signal_present, response, correct,
                         confidence, round(rt, 3), datetime.now().isoformat()])

    if trial % 25 == 0 and trial != n_trials:
        show_text_block(
            f"You may take a short break.\n\nYou've completed {trial} of {n_trials} trials.\n\nPress any key to continue."
        )
        event.waitKeys()

# ---------- END ----------
show_text_block("Thank you for participating!\n\nCalculating results...", duration=1.5)

# ---------- SDT Analysis ----------
n_signal = sum(1 for r in rows if r['signal_present'])
n_noise = len(rows) - n_signal
hits = sum(1 for r in rows if r['signal_present'] and r['response'] == 'yes')
fas = sum(1 for r in rows if not r['signal_present'] and r['response'] == 'yes')

H = hits / n_signal if n_signal else 0
F = fas / n_noise if n_noise else 0

def adjust_rate(rate, n):
    if rate >= 1.0:
        return (n - 0.5) / n
    if rate <= 0.0:
        return 0.5 / n
    return rate

H_adj = adjust_rate(H, n_signal)
F_adj = adjust_rate(F, n_noise)

if SCIPY_PRESENT:
    zH, zF = norm.ppf(H_adj), norm.ppf(F_adj)
    dprime = zH - zF
    c = -0.5 * (zH + zF)
else:
    dprime, c = "NA", "NA"

# ---------- ROC ----------
roc_rows = []
total_signal = sum(1 for r in rows if r['signal_present'])
total_noise = len(rows) - total_signal
for thresh in range(1, 6):
    hit_t = sum(1 for r in rows if r['signal_present'] and r['response'] == 'yes' and r['confidence'] >= thresh)
    fa_t = sum(1 for r in rows if not r['signal_present'] and r['response'] == 'yes' and r['confidence'] >= thresh)
    Ht = hit_t / total_signal
    Ft = fa_t / total_noise
    roc_rows.append({'Threshold': thresh, 'HitRate': round(Ht, 3), 'FARate': round(Ft, 3)})

# ---------- Binomial Test ----------
overall_correct = sum(r['correct'] for r in rows)
p_val = None
if SCIPY_PRESENT:
    try:
        res = binom_test(overall_correct, len(rows), 0.5, alternative='greater')
        p_val = res.pvalue if hasattr(res, 'pvalue') else res
    except Exception:
        p_val = None

# ---------- Write Summary ----------
with open(summaryfile, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["n_signal", "n_noise", "hits", "false_alarms", "H_rate", "FA_rate",
                     "dprime", "c", "overall_correct", "total_trials", "binomial_p"])
    writer.writerow([n_signal, n_noise, hits, fas, round(H, 4), round(F, 4), dprime, c,
                     overall_correct, len(rows), p_val])

with open(rocfile, "w", newline="") as rf:
    writer = csv.DictWriter(rf, fieldnames=['Threshold', 'HitRate', 'FARate'])
    writer.writeheader()
    for row in roc_rows:
        writer.writerow(row)

# ---------- Optional Plot ----------
try:
    import matplotlib.pyplot as plt
    F_vals = [r['FARate'] for r in roc_rows]
    H_vals = [r['HitRate'] for r in roc_rows]
    plt.figure()
    plt.plot(F_vals, H_vals, marker='o')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel("False Alarm Rate")
    plt.ylabel("Hit Rate")
    plt.title(f"ROC Curve - {participant}")
    plt.grid(True)
    plt.show()
except Exception as e:
    print("Plotting failed:", e)

# ---------- End Message ----------
res_lines = [
    f"Trials completed: {len(rows)}",
    f"Overall accuracy: {overall_correct}/{len(rows)} = {overall_correct/len(rows):.3f}",
    f"d': {dprime}, c: {c}",
]
if p_val is not None:
    res_lines.append(f"Binomial p (vs chance): {p_val:.4f}")
res_lines.append(f"\nData saved to:\n{trialfile}\n{summaryfile}\n{rocfile}")

show_text_block("\n".join(res_lines), duration=7, wait_for_keys=False)
core.wait(2.0)
win.close()
core.quit()
