# Multi-Probe Word Recognition Task (merged response version)
# 1–5 scale only: 1 = Sure NEW, 5 = Sure OLD
# Removes separate Y/N response but keeps full analysis, saving, ROC, etc.

from psychopy import visual, core, event, gui
import random, csv, os
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

# ---------- Utility ----------
def show_text(win, text, duration=None):
    msg = visual.TextStim(win, text=text, color="white", height=0.07, wrapWidth=1.5)
    msg.draw()
    win.flip()
    if duration:
        core.wait(duration)
    else:
        event.clearEvents()
        event.waitKeys()

# ---------- Participant Info ----------
expInfo = {'Participant_ID': '', 'SetSize': ['4','6','8','10']}
dlg = gui.DlgFromDict(expInfo, title='Multi-Probe Word Memory Task')
if not dlg.OK:
    core.quit()

participant_id = expInfo['Participant_ID']
set_size = int(expInfo['SetSize'])
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---------- Setup Window ----------
win = visual.Window(fullscr=True, color="black", units="norm")

# ---------- Word Pool ----------
words_pool = [
    "Stone","Butter","Book","Corner","Ticket","Teeth","Cream","Arm","Grass","Stand","Letter","Stick",
    "Seat","Green","Bone","Wicket","Mud","Leg","Leather","Belt","Statue","Milk","Strong","Puddle",
    "Rain","Socks","Lather","Sheet","Braces","Cricket","Hockey","Skeleton","Jacket","Sour","Lemon",
    "Emerald","Perception","Weakness","Shaft","Spine","Binder","Wonder","Birthday","Party",
    "Reputation","Ancestor","Oldage","Gear","Turbine","Horseshoe","Hallway","Comfortable","Contemporary",
    "Arch","Church","Dare","Ward","Paradox","Narrow","Volunteer","Health","Personal","Trench","Drench",
    "Cinema","Flood","Desert","Cluster","Tread","Advice","Sailor","Declaration","Table","Arrest",
    "Police","Tail","Snow","Huge","Waves","Orange","Disclose","Philosophy","Service","Hemisphere",
    "Sky","Imposter","Chaos","Block","Drown","Bounce","Linger","Ginger","Navy","Front","Symbol",
    "Cake","Reign","Confuse","Tribe","Detector","Collapse","Frown","Cutting","Simplicity","Swim",
    "Merit","Crime","Van","Truck","Pasta","Pizza","Festival","Shinning","Unicorn","Cave","Rattle",
    "Ink","Furniture","Massive","Devices","Hobbies","Pen","Dish","Dedication","Mouse","Reptile",
    "Privacy","Wrap","Wasp","Guest","Homeland","Pinned","Vault","Cafeteria","Matcha","Choked","Elbow",
    "Corpse","Monk","Chapter","Jury","Injury","Killer","Mood","Swing","Liability","Disability","Glacier",
    "Dignity","Relinquish","Sea","Feelings","Filling","Boy","Sick","Loan","Consumer","Basketball",
    "Curtains","Clarify","Tourists","Beaches","Palm","Suggest","Spotted","Software","Engineer",
    "Concede","Embedded","Brilliance","Galaxy","Railroad","Volcano","Ashes","Morsel","Permanent","Class"
]

# Normalize + remove duplicates
_words_normalized = [w.strip().capitalize() for w in words_pool]
seen = set()
words_pool_clean = []
for w in _words_normalized:
    if w not in seen:
        seen.add(w)
        words_pool_clean.append(w)
words_pool = words_pool_clean

# ---------- Parameters ----------
n_trials = 10
results = []
summary = {'H': 0, 'F': 0, 'target_trials': 0, 'distractor_trials': 0}

show_text(win,
    "Welcome!\nYou will see sets of words blinking on the screen.\n"
    "You will then be tested on your memory.\n\n"
    "For each probe, rate confidence directly:\n"
    "1 = Sure NEW (definitely not in list)\n"
    "2 = Probably NEW\n"
    "3 = Unsure\n"
    "4 = Probably OLD\n"
    "5 = Sure OLD (definitely in list)\n\nPress any key to begin."
)

used_words = set()
escape_pressed = False

# ---------- Experiment Loop ----------
for trial in range(1, n_trials + 1):
    avail_for_study = [w for w in words_pool if w not in used_words]
    if len(avail_for_study) < set_size:
        avail_for_study = words_pool.copy()
    study_set = random.sample(avail_for_study, set_size)

    # study phase
    for word in study_set:
        show_text(win, word, duration=0.8)
        core.wait(0.15)
    show_text(win, "+", duration=0.5)

    n_targets = set_size // 2
    n_lures = set_size - n_targets
    targets = random.sample(study_set, n_targets)

    lures_pool = [w for w in words_pool if w not in study_set and w not in used_words]
    if len(lures_pool) < n_lures:
        lures_pool = [w for w in words_pool if w not in study_set]
    lures = random.sample(lures_pool, n_lures)

    test_probes = [(w, True) for w in targets] + [(w, False) for w in lures]
    random.shuffle(test_probes)
    used_words.update(study_set)
    used_words.update(lures)

    for probe_word, is_target in test_probes:
        show_text(win, f"{probe_word}\n\nRate your confidence:\n1 = Sure NEW ... 5 = Sure OLD")
        event.clearEvents()
        key = event.waitKeys(keyList=['1','2','3','4','5','escape'])[0]
        if key == 'escape':
            show_text(win, "Experiment aborted. Results will be saved for completed trials.")
            escape_pressed = True
            break
        rating = int(key)

        # derive binary yes/no from rating
        resp_key = 'y' if rating >= 4 else 'n'
        correct = (is_target and resp_key == 'y') or (not is_target and resp_key == 'n')

        results.append({
            'Participant_ID': participant_id,
            'SetSize': set_size,
            'Trial': trial,
            'Probe': probe_word,
            'IsTarget': is_target,
            'Response': resp_key,
            'Confidence': rating,
            'Correct': int(correct)
        })

        if is_target:
            summary['target_trials'] += 1
            if resp_key == 'y': summary['H'] += 1
        else:
            summary['distractor_trials'] += 1
            if resp_key == 'y': summary['F'] += 1

        core.wait(0.15)

    if escape_pressed:
        break
    elif trial < n_trials:
        show_text(win, "Next trial...", duration=0.6)

# ---------- Wrap-up ----------
show_text(win, "Experiment complete!\nPress any key to continue.")
event.clearEvents(); event.waitKeys(); win.close()

# ---------- Compute d′ and c ----------
H = summary['H'] / summary['target_trials'] if summary['target_trials'] else 0.0
F = summary['F'] / summary['distractor_trials'] if summary['distractor_trials'] else 0.0

if summary['target_trials']:
    H = max(min(H, 1 - 1/(2*summary['target_trials'])), 1/(2*summary['target_trials']))
if summary['distractor_trials']:
    F = max(min(F, 1 - 1/(2*summary['distractor_trials'])), 1/(2*summary['distractor_trials']))

try:
    d_prime = norm.ppf(H) - norm.ppf(F)
    c = -0.5 * (norm.ppf(H) + norm.ppf(F))
except Exception:
    d_prime, c = float('nan'), float('nan')

# ---------- ROC ----------
total_targets = summary['target_trials'] or 1
total_distractors = summary['distractor_trials'] or 1

roc_table = []
for thresh in range(1, 6):
    hits_t = sum(1 for r in results if r['IsTarget'] and r['Confidence'] >= thresh)
    fas_t  = sum(1 for r in results if (not r['IsTarget']) and r['Confidence'] >= thresh)
    Ht, Ft = hits_t/total_targets, fas_t/total_distractors
    roc_table.append({'Threshold': thresh, 'HitRate': round(Ht,3), 'FARate': round(Ft,3)})

# ---------- Save CSVs ----------
main_filename = f"WM_{participant_id}_Set{set_size}_{timestamp}.csv"
with open(main_filename, 'w', newline='') as mf:
    fieldnames = ['Participant_ID','SetSize','Trial','Probe','IsTarget','Response','Confidence','Correct']
    writer = csv.DictWriter(mf, fieldnames=fieldnames)
    writer.writeheader()
    for row in results: writer.writerow(row)
    writer.writerow({})
    writer.writerow({'Participant_ID':'HitRate','SetSize':round(H,3),'Trial':'FARate','Probe':round(F,3)})
    writer.writerow({'Participant_ID':"d'",'SetSize':round(d_prime,3),'Trial':'c','Probe':round(c,3)})

roc_filename = f"ROC_Summary_{participant_id}_Set{set_size}_{timestamp}.csv"
with open(roc_filename, 'w', newline='') as rf:
    writer = csv.DictWriter(rf, fieldnames=['Threshold','HitRate','FARate'])
    writer.writeheader(); [writer.writerow(r) for r in roc_table]

# ---------- Plot ROC ----------
F_vals = [r['FARate'] for r in roc_table]
H_vals = [r['HitRate'] for r in roc_table]

plt.figure()
plt.plot(F_vals, H_vals, marker='o')
plt.plot([0,1],[0,1],'--')
plt.xlim(0,1); plt.ylim(0,1)
plt.xlabel("False Alarm Rate"); plt.ylabel("Hit Rate")
plt.title(f"ROC Curve - Participant {participant_id} (Set size {set_size})")
plt.grid(True)
plt.show()

print("Experiment finished. You may close this window.")
core.quit()
