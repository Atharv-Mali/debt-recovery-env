"""Generate 500 synthetic Indian NBFC borrower profiles."""
import json
import random

random.seed(2024)

FIRST_NAMES_MALE = [
    "Rajesh", "Amit", "Suresh", "Vikram", "Arun", "Sanjay", "Manoj", "Deepak",
    "Ravi", "Ashok", "Ganesh", "Pramod", "Sunil", "Vinod", "Ramesh", "Nitin",
    "Ajay", "Rahul", "Kiran", "Sachin", "Naveen", "Mohan", "Gopal", "Dinesh",
    "Anand", "Pankaj", "Mukesh", "Harish", "Jitendra", "Rakesh", "Yogesh",
    "Mahesh", "Naresh", "Shyam", "Rajendra", "Vijay", "Kamal", "Girish",
    "Satish", "Pradeep", "Balaji", "Srinivas", "Venkatesh", "Raghav", "Arjun",
    "Manish", "Rohit", "Sandeep", "Anil", "Bharat", "Chandra", "Dev", "Gaurav",
    "Hemant", "Iqbal", "Jagdish", "Kishore", "Lalit", "Madhav", "Narayan",
    "Om", "Prakash", "Qasim", "Rustam", "Shankar", "Trilok", "Umesh",
    "Varun", "Wasim", "Yashwant", "Zaheer"
]

FIRST_NAMES_FEMALE = [
    "Priya", "Sunita", "Anita", "Kavita", "Rekha", "Meena", "Seema", "Pooja",
    "Neha", "Asha", "Geeta", "Savita", "Lata", "Usha", "Suman", "Kalpana",
    "Shanti", "Parvati", "Sarita", "Mamta", "Jaya", "Radha", "Nirmala",
    "Kamala", "Padma", "Indira", "Leela", "Durga", "Shakuntala", "Vijaya",
    "Deepa", "Renu", "Swati", "Archana", "Bhavna", "Chhaya", "Divya",
    "Ekta", "Farida", "Gauri", "Hema", "Ila", "Jyoti", "Komal", "Lalita",
    "Manju", "Nandini", "Pallavi", "Rashmi", "Shalini", "Tanvi", "Uma",
    "Vandana", "Yamini", "Zeenat"
]

LAST_NAMES = [
    "Kumar", "Sharma", "Singh", "Verma", "Gupta", "Patel", "Joshi", "Reddy",
    "Nair", "Iyer", "Rao", "Das", "Mehta", "Shah", "Mishra", "Pandey",
    "Tiwari", "Dubey", "Yadav", "Chauhan", "Thakur", "Bhat", "Menon",
    "Pillai", "Desai", "Patil", "Kulkarni", "Jain", "Agarwal", "Banerjee",
    "Mukherjee", "Chatterjee", "Bose", "Sen", "Ghosh", "Roy", "Dutta",
    "Malhotra", "Kapoor", "Khanna", "Saxena", "Srivastava", "Bajaj",
    "Chowdhury", "Biswas", "Sethi", "Bhatia", "Arora", "Choudhary", "Goyal",
    "Prasad", "Rajan", "Naidu", "Hegde", "Shetty", "Kamath", "Pai",
    "Gowda", "Swamy", "Mohan"
]

EMPLOYMENT_TYPES = ["salaried", "self_employed", "daily_wage", "unemployed"]
EMPLOYMENT_WEIGHTS = [0.40, 0.30, 0.20, 0.10]

INCOME_BANDS = ["low", "low_mid", "mid", "high"]
PRODUCT_TYPES = ["personal_loan", "auto_loan", "home_loan", "microfinance"]

SENTIMENTS = ["cooperative", "avoidant", "hostile", "ghost"]
LEGAL_STAGES = ["pre_legal", "notice_sent", "sarfaesi", "drt", "written_off"]

profiles = []

for i in range(500):
    profile_id = f"BRW_{i+1:03d}"

    if random.random() < 0.55:
        first = random.choice(FIRST_NAMES_MALE)
    else:
        first = random.choice(FIRST_NAMES_FEMALE)
    last = random.choice(LAST_NAMES)
    name = f"{first} {last}"

    employment = random.choices(EMPLOYMENT_TYPES, weights=EMPLOYMENT_WEIGHTS, k=1)[0]

    # DPD distribution: skewed toward lower values
    dpd = int(random.betavariate(2, 5) * 175) + 5
    dpd = max(5, min(180, dpd))

    # Outstanding amount correlates with product type
    product = random.choice(PRODUCT_TYPES)
    if product == "home_loan":
        outstanding = random.randint(100000, 500000)
    elif product == "auto_loan":
        outstanding = random.randint(50000, 300000)
    elif product == "personal_loan":
        outstanding = random.randint(8000, 200000)
    else:  # microfinance
        outstanding = random.randint(8000, 50000)

    # Credit score: lower for higher DPD
    base_credit = random.randint(550, 780)
    credit_score = max(450, base_credit - int(dpd * 0.8))

    # Income band correlates with employment
    if employment == "salaried":
        income_band = random.choices(INCOME_BANDS, weights=[0.1, 0.3, 0.4, 0.2], k=1)[0]
    elif employment == "self_employed":
        income_band = random.choices(INCOME_BANDS, weights=[0.15, 0.35, 0.35, 0.15], k=1)[0]
    elif employment == "daily_wage":
        income_band = random.choices(INCOME_BANDS, weights=[0.5, 0.35, 0.12, 0.03], k=1)[0]
    else:
        income_band = random.choices(INCOME_BANDS, weights=[0.7, 0.2, 0.08, 0.02], k=1)[0]

    city_tier = random.choices([1, 2, 3], weights=[0.25, 0.40, 0.35], k=1)[0]

    # Hardship flag: correlates with daily_wage/unemployed and high DPD
    hardship_base = 0.05
    if employment in ("daily_wage", "unemployed"):
        hardship_base += 0.35
    if dpd > 90:
        hardship_base += 0.15
    if income_band == "low":
        hardship_base += 0.10
    hardship_flag = random.random() < hardship_base

    # Sentiment: ghost correlates with DPD>120, hostile with high DPD
    if dpd > 120:
        sentiment = random.choices(SENTIMENTS, weights=[0.05, 0.20, 0.25, 0.50], k=1)[0]
    elif dpd > 90:
        sentiment = random.choices(SENTIMENTS, weights=[0.10, 0.30, 0.35, 0.25], k=1)[0]
    elif dpd > 60:
        sentiment = random.choices(SENTIMENTS, weights=[0.20, 0.40, 0.25, 0.15], k=1)[0]
    elif dpd > 30:
        sentiment = random.choices(SENTIMENTS, weights=[0.35, 0.40, 0.15, 0.10], k=1)[0]
    else:
        sentiment = random.choices(SENTIMENTS, weights=[0.55, 0.30, 0.10, 0.05], k=1)[0]

    # Legal stage correlates with DPD
    if dpd < 30:
        legal_stage = "pre_legal"
    elif dpd < 60:
        legal_stage = random.choices(["pre_legal", "notice_sent"], weights=[0.7, 0.3], k=1)[0]
    elif dpd < 90:
        legal_stage = random.choices(["pre_legal", "notice_sent", "sarfaesi"], weights=[0.3, 0.5, 0.2], k=1)[0]
    elif dpd < 120:
        legal_stage = random.choices(["notice_sent", "sarfaesi", "drt"], weights=[0.3, 0.4, 0.3], k=1)[0]
    else:
        legal_stage = random.choices(["sarfaesi", "drt", "written_off"], weights=[0.3, 0.4, 0.3], k=1)[0]

    # PTP history: broken correlates with high DPD
    ptp_made = random.randint(0, 5)
    if dpd > 90:
        broken_ratio = random.uniform(0.5, 0.9)
    elif dpd > 60:
        broken_ratio = random.uniform(0.3, 0.6)
    else:
        broken_ratio = random.uniform(0.0, 0.4)
    ptp_broken = min(ptp_made, int(ptp_made * broken_ratio))
    ptp_kept = ptp_made - ptp_broken

    # Days since last payment correlates with DPD
    days_since_last_payment = max(0, dpd + random.randint(-10, 20))

    # DNC
    dnc_registered = random.random() < 0.08

    # Complaints correlate with hostile sentiment
    complaint_count = 0
    if sentiment == "hostile":
        complaint_count = random.choices([0, 1, 2, 3], weights=[0.4, 0.3, 0.2, 0.1], k=1)[0]
    elif sentiment == "avoidant":
        complaint_count = random.choices([0, 1], weights=[0.85, 0.15], k=1)[0]

    reference_contact_available = random.random() < 0.65

    profile = {
        "id": profile_id,
        "name": name,
        "outstanding_inr": outstanding,
        "dpd": dpd,
        "credit_score": credit_score,
        "employment_type": employment,
        "income_band": income_band,
        "city_tier": city_tier,
        "hardship_flag": hardship_flag,
        "legal_stage": legal_stage,
        "ptp_history": {
            "made": ptp_made,
            "kept": ptp_kept,
            "broken": ptp_broken
        },
        "sentiment": sentiment,
        "product_type": product,
        "days_since_last_payment": days_since_last_payment,
        "reference_contact_available": reference_contact_available,
        "dnc_registered": dnc_registered,
        "complaint_count": complaint_count
    }
    profiles.append(profile)

with open("data/borrower_profiles.json", "w", encoding="utf-8") as f:
    json.dump(profiles, f, indent=2, ensure_ascii=False)

print(f"Generated {len(profiles)} borrower profiles.")

# Print distribution stats
from collections import Counter
print(f"\nSentiment distribution: {Counter(p['sentiment'] for p in profiles)}")
print(f"Employment distribution: {Counter(p['employment_type'] for p in profiles)}")
print(f"Legal stage distribution: {Counter(p['legal_stage'] for p in profiles)}")
print(f"DPD range: {min(p['dpd'] for p in profiles)} - {max(p['dpd'] for p in profiles)}")
print(f"Outstanding range: {min(p['outstanding_inr'] for p in profiles)} - {max(p['outstanding_inr'] for p in profiles)}")
print(f"Hardship flagged: {sum(1 for p in profiles if p['hardship_flag'])}")
print(f"DNC registered: {sum(1 for p in profiles if p['dnc_registered'])}")
