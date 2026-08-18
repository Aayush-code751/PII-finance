"""SynPII-FH generator: 6 templates (3 finance, 3 clinical), 11 entity types,
controlled duplicate injection. Gold spans recorded at generation time.
Entity phrasing intentionally varies anchored/unanchored PERSON mentions.
Also builds the near-duplicate STRESS set: same template, different entities.
"""
from __future__ import annotations
import random, re
from .core import Span, _luhn_check_digit, iban_valid

F = ["James","Maria","Wei","Aisha","Carlos","Yuki","Fatima","Ivan","Priya","Liam",
     "Sofia","Noah","Amara","Diego","Hana","Omar","Elena","Kofi","Ingrid","Ravi",
     "Lucia","Ahmed","Greta","Mateo","Nadia","Oscar","Mei","Tomas","Zara","Felix"]
L = ["Okafor","Lindqvist","Marchetti","Osei","Petrov","Tanaka","Alvarez","Novak",
     "Sharma","Keller","Bianchi","Haddad","Larsen","Moreau","Silva","Kowalski",
     "Nguyen","Fischer","Romano","Ivanov","Costa","Berg","Weber","Sato","Khan"]
STREET_N = ["Oakwood","Riverside","Sunset","Highland","Meadow","Franklin","Jefferson",
            "Chestnut","Sycamore","Magnolia","Colonial","Prospect","Sherman","Bayview"]
SUF = ["St","Ave","Rd","Ln","Blvd","Dr","Ct","Way"]
CITY = [("Springfield","IL"),("Riverton","NJ"),("Lakewood","CA"),("Fairview","TX"),
        ("Georgetown","KY"),("Bristol","CT"),("Clayton","MO"),("Ashland","OR")]
DIAG = ["type 2 diabetes mellitus","atrial fibrillation","community-acquired pneumonia",
        "chronic kidney disease stage 3","major depressive disorder","asthma exacerbation"]
MEDS = ["metformin 500 mg BID","apixaban 5 mg BID","lisinopril 10 mg daily",
        "atorvastatin 40 mg nightly","sertraline 50 mg daily","albuterol PRN"]
PURPOSE = ["invoice settlement","equipment purchase","escrow funding",
           "supplier payment","tuition payment","real estate closing"]

def _v(rng):
    """One entity bundle with char-exact values."""
    person = f"{rng.choice(F)} {rng.choice(L)}"
    street = f"{rng.randint(10,9899)} {rng.choice(STREET_N)} {rng.choice(SUF)}"
    city, st = rng.choice(CITY)
    address = f"{street}, {city}, {st} {rng.randint(10000,99999)}"
    body = "".join(str(rng.randint(0,9)) for _ in range(15))
    card_digits = body + _luhn_check_digit(body)
    card = " ".join(card_digits[i:i+4] for i in range(0,16,4))
    ib_body = "".join(str(rng.randint(0,9)) for _ in range(16))
    iban = next(f"DE{c:02d}{ib_body}" for c in range(2,99) if iban_valid(f"DE{c:02d}{ib_body}"))
    return dict(
        PERSON=person,
        EMAIL=f"{person.split()[0].lower()}.{person.split()[1].lower()}@{rng.choice(['mailbox.example','corpmail.example','clinicmail.example'])}.com",
        PHONE=f"({rng.randint(201,989)}) {rng.randint(200,989)}-{rng.randint(1000,9999)}",
        SSN=f"{rng.randint(100,665):03d}-{rng.randint(1,99):02d}-{rng.randint(1,9999):04d}",
        DOB=f"{rng.randint(1,12):02d}/{rng.randint(1,28):02d}/{rng.randint(1938,2004)}",
        ADDRESS=address, CREDIT_CARD=card, IBAN=iban,
        ACCOUNT_NUMBER=str(rng.randint(10**9, 10**12)),
        MRN=f"{rng.choice('ABCDEFGH')}{rng.randint(10**5,10**7)}",
        INSURANCE_ID=f"{rng.choice(['BCX','AET','UHG','CIG'])}-{rng.randint(10**6,10**8)}",
    )

# Each template: (domain, fn(v, rng) -> text with {TYPE:value} markers inline)
def t_kyc(v, rng):
    return (f"KYC ONBOARDING SUMMARY\nApplicant: [PERSON]. Verified residence at [ADDRESS]. "
            f"Contact phone [PHONE], email [EMAIL].\nGovernment identifier (SSN): [SSN]. "
            f"Date of birth: [DOB].\nPrimary funding card [CREDIT_CARD]; settlement IBAN [IBAN]. "
            f"Core account number: [ACCOUNT_NUMBER].\nRisk tier assigned after enhanced due diligence. "
            f"[PERSON2] of compliance countersigned the file.")

def t_wire(v, rng):
    return (f"WIRE TRANSFER MEMO\nOn behalf of [PERSON], remit funds for {rng.choice(PURPOSE)}. "
            f"Debit account [ACCOUNT_NUMBER]; beneficiary IBAN [IBAN].\n"
            f"Callback phone on file: [PHONE]. Confirmation to [EMAIL].\n"
            f"Card on file for fees: [CREDIT_CARD]. Sender residence: [ADDRESS]. "
            f"Authorized {rng.choice(['am','pm'])} by desk officer [PERSON2].")

def t_loan(v, rng):
    return (f"LOAN NOTE ADDENDUM\nBorrower [PERSON], DOB [DOB], SSN [SSN], residing at [ADDRESS]. "
            f"Disbursement to account [ACCOUNT_NUMBER] (backup IBAN [IBAN]).\n"
            f"Autopay card: [CREDIT_CARD]. Servicing contact: [PHONE] / [EMAIL].\n"
            f"Cosigner [PERSON2] executed remotely.")

def t_discharge(v, rng):
    return (f"DISCHARGE SUMMARY\nPatient: [PERSON] (MRN [MRN]). DOB [DOB]. "
            f"Admitted for {rng.choice(DIAG)}.\nInsurance member ID: [INSURANCE_ID]. "
            f"Home address [ADDRESS]. Callback [PHONE]; portal email [EMAIL].\n"
            f"Discharge medications: {rng.choice(MEDS)}; {rng.choice(MEDS)}.\n"
            f"Follow-up arranged by [PERSON2], care coordinator. SSN on billing file: [SSN].")

def t_progress(v, rng):
    return (f"PROGRESS NOTE\n[PERSON2] documented overnight events. Patient [PERSON], "
            f"medical record no. [MRN], stable on {rng.choice(MEDS)}.\n"
            f"Insurance ID: [INSURANCE_ID]. Reached emergency contact at [PHONE]. "
            f"Demographics on file: DOB [DOB], address [ADDRESS], email [EMAIL].")

def t_referral(v, rng):
    return (f"REFERRAL LETTER\nRe: [PERSON], DOB [DOB], MRN [MRN].\n"
            f"Dear colleague, I am referring this patient for evaluation of {rng.choice(DIAG)}. "
            f"Member ID [INSURANCE_ID]. The family can be reached at [PHONE] or [EMAIL]; "
            f"they reside at [ADDRESS].\nKind regards, Dr. [PERSON2]")

TEMPLATES = [("finance", t_kyc), ("finance", t_wire), ("finance", t_loan),
             ("healthcare", t_discharge), ("healthcare", t_progress), ("healthcare", t_referral)]

def _materialize(tpl_fn, rng):
    v = _v(rng); v2 = _v(rng)
    text = tpl_fn(v, rng)
    gold: list[Span] = []
    def repl(m):
        key = m.group(1)
        val = v2["PERSON"] if key == "PERSON2" else v[key]
        gold.append(Span(len(out[0]), len(out[0]) + len(val),
                         "PERSON" if key == "PERSON2" else key, val))
        out[0] += val
        return ""
    out = [""]
    pos = 0
    for m in re.finditer(r"\[([A-Z_2]+)\]", text):
        out[0] += text[pos:m.start()]; repl(m); pos = m.end()
    out[0] += text[pos:]
    return out[0], gold

def generate_corpus(n_unique: int = 600, dup_frac: float = 0.12, seed: int = 7):
    """Returns list of dicts {id,text,gold,domain,is_dup,dup_kind,src_id}."""
    rng = random.Random(seed)
    docs = []
    for i in range(n_unique):
        domain, fn = TEMPLATES[i % len(TEMPLATES)]
        text, gold = _materialize(fn, rng)
        docs.append(dict(id=f"u{i}", text=text, gold=gold, domain=domain,
                         is_dup=False, dup_kind="", src_id=""))
    n_dup = int(round(n_unique * dup_frac / (1 - dup_frac)))
    dups = []
    for j in range(n_dup):
        src = rng.choice(docs)
        if j % 2 == 0:
            dups.append(dict(id=f"d{j}", text=src["text"], gold=list(src["gold"]),
                             domain=src["domain"], is_dup=True, dup_kind="exact",
                             src_id=src["id"]))
        else:
            foot = f"\n-- routing: desk {rng.randint(1,9)} / batch {rng.randint(100,999)} --"
            txt = re.sub(r" ", lambda m: "  " if rng.random() < 0.06 else " ",
                         src["text"]) + foot
            dups.append(dict(id=f"d{j}", text=txt, gold=None, domain=src["domain"],
                             is_dup=True, dup_kind="near", src_id=src["id"]))
    stream = docs + dups
    rng.shuffle(stream)
    return stream

def stress_pairs(n_pairs: int = 150, seed: int = 7, minimal: bool = False):
    """Same template, DIFFERENT entities: must NOT be suppressed.
    minimal=True is the harshest condition: filler choices (diagnoses, meds,
    purposes) are held IDENTICAL and only the entity values differ."""
    rng = random.Random(seed * 31 + 5)
    pairs = []
    for i in range(n_pairs):
        _, fn = TEMPLATES[i % len(TEMPLATES)]
        if minimal:
            fseed = rng.randrange(1 << 30)
            va, vb = _v(rng), _v(rng)
            a = _fill(fn, va, random.Random(fseed))
            b = _fill(fn, vb, random.Random(fseed))
        else:
            a, _ = _materialize(fn, rng)
            b, _ = _materialize(fn, rng)
        pairs.append((a, b))
    return pairs

def _fill(tpl_fn, v, rng):
    """Materialize template with a given entity bundle and filler rng."""
    v2 = dict(v); v2["PERSON"] = f"{rng.choice(F)} {rng.choice(L)}"
    text = tpl_fn(v, rng)
    out, pos = "", 0
    for m in re.finditer(r"\[([A-Z_2]+)\]", text):
        key = m.group(1)
        val = v2["PERSON"] if key == "PERSON2" else v[key]
        out += text[pos:m.start()] + val; pos = m.end()
    return out + text[pos:]
