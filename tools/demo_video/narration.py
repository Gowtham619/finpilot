"""Voice-over lines. Each key is either a scene marker set by build.py or a regex matched against the tour's captions."""

VOICE, RATE = "Tara", 172

# (key, text). Keys starting with "cap:" match the Guided demo caption at the moment it appears.
LINES = [
    ("intro", "FinPilot is a financial copilot that shows you the weather, not just the balance. "
              "Give it a bank statement, and a team of agents categorizes it, finds problems, simulates a thousand possible futures, "
              "and forecasts your next sixty days."),
    ("import_open", "It starts with Import statement. FinPilot reads CSV files, text PDFs, and even scans or photos of a statement. "
                    "Everything is read on this machine, and nothing is sent to an outside service."),
    ("import_scan", "Here is a scanned page. Local OCR reads it, and the app tells you when digits might have been misread."),
    ("import_text", "Now a text PDF. Before anything is saved, you get a preview: how many transactions, the date range, money in and out, "
                    "and the opening balance, which it found by itself in the balance column. "
                    "Every row is checked against the running balance. And look at the descriptions: names, phone numbers, UPI IDs and account numbers "
                    "are already masked."),
    ("import_confirm", "Confirm, and the agents get to work."),
    ("import_done", "The statement now has its own forecast, with the first findings underneath."),
    ("tour_intro", "Now the built in demo persona, Priya, with twelve months of synthetic transactions. "
                   "Watch the trace on the right: the categorizer, analyst, forecaster, advisor and verifier each report as they finish."),
    ("cap:^Balance", "The dashboard opens with a forecast headline and a sixty day weather strip, one block for each day. "
                     "Below it are the balance, the median outcome in sixty days, and the chance of an overdraft."),
    ("cap:^The fan chart", "The chart is the plausible range for your balance. The middle line is the median, and the band is the ten to ninety percent range, "
                           "from a thousand simulated futures built from your recent habits and recurring bills."),
    ("cap:^Money Weather, Normal", "Money Weather turns the same simulation into a risk calendar for the next sixty days. Priya is fine today, so the skies are clear."),
    ("cap:^Now a big purchase", "Now stress it. A ninety four thousand rupee purchase on October tenth, and most simulated futures dip below the safety buffer."),
    ("cap:^The autopsy", "The autopsy compares the failing futures with the fine ones, to show which habits and bills drive the risk."),
    ("cap:^An umbrella", "An umbrella is a spending cut, ranked by how much risk it removes for each thousand rupees you give up."),
    ("cap:^With the umbrella", "Applying it lowers the chance of trouble."),
    ("cap:^Non-bill spending", "And this slider shows how much trimming everyday spending changes the whole picture."),
    ("cap:^Why\\?", "Every recommendation can show its evidence, and what the verifier recomputed from your own transactions."),
    ("cap:^Can I afford", "Can I afford it simulates a purchase against the same futures, and suggests better dates."),
    ("cap:^Decision duel", "The decision duel runs two options on identical random futures, and names a winner with numbers."),
    ("cap:^Finally the digest", "A weekly digest summarizes the findings. It is stored, and emailed only if mail is configured."),
    ("weather_jobloss", "Back in Money Weather, the job loss drill removes your next paychecks, to see how long your savings hold."),
    ("weather_medical", "The medical emergency scenario drops a sudden expense into every future."),
    ("recs", "Recommendations list each action in plain words, like cancelling an unused subscription or disputing a duplicate charge. "
             "The Why panel shows the exact figures used."),
    ("goals", "Goals track a target balance by a deadline, and tell you the chance of reaching it."),
    ("chat", "And you can ask questions about your own spending in plain language. Answers come from read only queries on your data, not guesses."),
    ("outro", "That is FinPilot. The engines do the arithmetic, the language model only explains, and a verifier checks every number. "
              "All data in this video is synthetic."),
]
