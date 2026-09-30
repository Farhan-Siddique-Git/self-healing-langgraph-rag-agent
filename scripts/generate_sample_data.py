"""
Generates synthetic sample data for Stage 0 of the Self-Healing Document RAG Agent:
  - A handful of realistic-looking HR policy PDFs (data/policies/*.pdf)
  - An employees.csv roster (data/employees.csv)

This uses only matplotlib + stdlib csv (both pre-installed almost everywhere),
so it has no extra dependencies beyond what's in requirements.txt.

Run:  python scripts/generate_sample_data.py
"""
import csv
import os
import textwrap

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES_DIR = os.path.join(HERE, "data", "policies")
EMPLOYEES_CSV = os.path.join(HERE, "data", "employees.csv")

os.makedirs(POLICIES_DIR, exist_ok=True)


def write_pdf(filename: str, title: str, sections: list[tuple[str, str]]) -> None:
    """Render a simple text-only policy document as a multi-page PDF."""
    path = os.path.join(POLICIES_DIR, filename)
    with PdfPages(path) as pdf:
        # Wrap each section's body into lines that fit the page.
        page_lines: list[str] = [title, ""]
        for heading, body in sections:
            page_lines.append(heading)
            page_lines.append("-" * len(heading))
            wrapped = textwrap.wrap(body, width=95)
            page_lines.extend(wrapped)
            page_lines.append("")

        lines_per_page = 45
        for start in range(0, len(page_lines), lines_per_page):
            chunk = page_lines[start : start + lines_per_page]
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(
                0.08,
                0.95,
                "\n".join(chunk),
                fontsize=10,
                family="monospace",
                verticalalignment="top",
            )
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Policy documents
# ---------------------------------------------------------------------------

write_pdf(
    "remote_work_policy.pdf",
    "Acme Corp — Remote Work Policy (Effective Jan 2026)",
    [
        (
            "1. Eligibility",
            "Full-time employees who have completed their 90-day probationary period are "
            "eligible to work remotely up to 3 days per week, subject to manager approval. "
            "Interns and part-time contractors are NOT eligible for remote work under this "
            "policy and are expected to work on-site for the duration of their placement, "
            "unless an accommodation has been separately approved by HR.",
        ),
        (
            "2. Approval Process",
            "Employees must submit a remote work request through the HR portal at least 5 "
            "business days in advance. Managers are expected to respond within 2 business "
            "days. Standing remote arrangements (recurring weekly days) must be renewed "
            "every 6 months.",
        ),
        (
            "3. Equipment",
            "Acme Corp provides a standard remote work stipend of 250 EUR per year for home "
            "office equipment. Employees are responsible for maintaining a secure, private "
            "internet connection and must use the company VPN when accessing internal "
            "systems remotely.",
        ),
        (
            "4. Core Hours",
            "All remote employees must be reachable via Slack and available for meetings "
            "between 10:00 and 16:00 CET, regardless of their chosen remote work days.",
        ),
    ],
)

write_pdf(
    "pto_policy.pdf",
    "Acme Corp — Paid Time Off (PTO) Policy",
    [
        (
            "1. Accrual",
            "Full-time employees accrue 2.083 PTO days per month, totaling 25 days per "
            "calendar year. PTO begins accruing from the first day of employment but "
            "cannot be used until after the 90-day probationary period ends.",
        ),
        (
            "2. Carryover",
            "Employees may carry over a maximum of 5 unused PTO days into the following "
            "calendar year. Any unused balance beyond 5 days is forfeited on December 31st "
            "unless local law requires otherwise.",
        ),
        (
            "3. Requesting Time Off",
            "PTO requests must be submitted through the HR portal at least 2 weeks in "
            "advance for requests longer than 3 consecutive days. Requests of 1-2 days may "
            "be submitted with 48 hours notice, manager approval permitting.",
        ),
        (
            "4. Sick Leave",
            "Sick leave is tracked separately from PTO and is not capped, subject to a "
            "medical certificate being provided for absences longer than 3 consecutive "
            "days.",
        ),
    ],
)

write_pdf(
    "expense_policy.pdf",
    "Acme Corp — Travel & Expense Policy",
    [
        (
            "1. Reimbursable Expenses",
            "Reasonable expenses for business travel, client meetings, and approved "
            "conferences are reimbursable, including economy-class flights, standard hotel "
            "accommodation, ground transportation, and meals up to 60 EUR per day.",
        ),
        (
            "2. Submission",
            "All expense claims must be submitted within 30 days of the expense being "
            "incurred, through the finance portal, with itemized receipts attached. Claims "
            "submitted after 60 days will not be reimbursed except in exceptional "
            "circumstances approved by Finance.",
        ),
        (
            "3. Non-Reimbursable Items",
            "Alcohol (outside of approved client entertainment), personal entertainment, "
            "traffic fines, and upgrades to business or first class travel are not "
            "reimbursable without prior VP-level approval.",
        ),
        (
            "4. Approval Thresholds",
            "Expense reports under 500 EUR require manager approval only. Reports between "
            "500 and 2000 EUR require manager and department head approval. Reports over "
            "2000 EUR require Finance approval in addition.",
        ),
    ],
)

write_pdf(
    "code_of_conduct.pdf",
    "Acme Corp — Code of Conduct",
    [
        (
            "1. Professional Behavior",
            "All employees, contractors, and interns are expected to treat colleagues, "
            "clients, and partners with respect and professionalism. Harassment, "
            "discrimination, and retaliation of any kind are strictly prohibited and will "
            "result in disciplinary action up to and including termination.",
        ),
        (
            "2. Conflicts of Interest",
            "Employees must disclose any outside business interests, board memberships, or "
            "financial relationships that could reasonably be seen as a conflict of "
            "interest with Acme Corp's business, using the annual disclosure form.",
        ),
        (
            "3. Confidentiality",
            "Employees must not share confidential company information, including "
            "unreleased product plans, customer data, or financial results, with anyone "
            "outside the company without written authorization.",
        ),
        (
            "4. Reporting Violations",
            "Suspected violations of this Code of Conduct should be reported to HR or "
            "through the anonymous ethics hotline. Retaliation against anyone reporting in "
            "good faith is strictly prohibited.",
        ),
    ],
)

write_pdf(
    "security_policy.pdf",
    "Acme Corp — Information Security Policy",
    [
        (
            "1. Account Security",
            "All employees must use multi-factor authentication (MFA) on their corporate "
            "accounts and a company-approved password manager. Passwords must not be "
            "reused across personal and corporate accounts.",
        ),
        (
            "2. Device Management",
            "Company laptops must have disk encryption and endpoint protection software "
            "enabled at all times. Personal devices used to access corporate email must be "
            "enrolled in the mobile device management (MDM) system.",
        ),
        (
            "3. Data Classification",
            "Data is classified as Public, Internal, Confidential, or Restricted. "
            "Restricted data (e.g. customer PII, financial records) may only be stored in "
            "approved systems and must never be emailed or copied to personal devices.",
        ),
        (
            "4. Incident Reporting",
            "Any suspected security incident, including lost devices, phishing attempts, "
            "or unauthorized access, must be reported to the Security team within 24 hours "
            "via the #security-incidents Slack channel or security@acmecorp.example.",
        ),
    ],
)

print(f"\n{5} policy PDFs written to {POLICIES_DIR}")

# ---------------------------------------------------------------------------
# Employee roster (DuckDB will load this CSV directly)
# ---------------------------------------------------------------------------

employees = [
    {"id": 1, "name": "Alice Nakamura", "department": "Engineering", "title": "Senior Software Engineer", "manager": "David Reyes", "email": "alice.nakamura@acmecorp.example", "location": "Magdeburg, DE", "employment_type": "Full-time"},
    {"id": 2, "name": "Bilal Ahmed", "department": "Engineering", "title": "Software Engineer", "manager": "David Reyes", "email": "bilal.ahmed@acmecorp.example", "location": "Berlin, DE", "employment_type": "Full-time"},
    {"id": 3, "name": "Chloe Fischer", "department": "Data Science", "title": "ML Engineer", "manager": "Priya Ramachandran", "email": "chloe.fischer@acmecorp.example", "location": "Magdeburg, DE", "employment_type": "Full-time"},
    {"id": 4, "name": "David Reyes", "department": "Engineering", "title": "Engineering Manager", "manager": "Sofia Bianchi", "email": "david.reyes@acmecorp.example", "location": "Magdeburg, DE", "employment_type": "Full-time"},
    {"id": 5, "name": "Priya Ramachandran", "department": "Data Science", "title": "Data Science Manager", "manager": "Sofia Bianchi", "email": "priya.ramachandran@acmecorp.example", "location": "Remote - EU", "employment_type": "Full-time"},
    {"id": 6, "name": "Sofia Bianchi", "department": "Engineering", "title": "VP of Engineering", "manager": "", "email": "sofia.bianchi@acmecorp.example", "location": "Berlin, DE", "employment_type": "Full-time"},
    {"id": 7, "name": "Tom Schneider", "department": "Engineering", "title": "Software Engineering Intern", "manager": "David Reyes", "email": "tom.schneider@acmecorp.example", "location": "Magdeburg, DE", "employment_type": "Intern"},
    {"id": 8, "name": "Maria Gonzalez", "department": "HR", "title": "HR Business Partner", "manager": "Jonas Weber", "email": "maria.gonzalez@acmecorp.example", "location": "Berlin, DE", "employment_type": "Full-time"},
    {"id": 9, "name": "Jonas Weber", "department": "HR", "title": "Head of People", "manager": "", "email": "jonas.weber@acmecorp.example", "location": "Berlin, DE", "employment_type": "Full-time"},
    {"id": 10, "name": "Elena Petrova", "department": "Finance", "title": "Financial Analyst", "manager": "Marcus Lindqvist", "email": "elena.petrova@acmecorp.example", "location": "Remote - EU", "employment_type": "Full-time"},
    {"id": 11, "name": "Marcus Lindqvist", "department": "Finance", "title": "Head of Finance", "manager": "", "email": "marcus.lindqvist@acmecorp.example", "location": "Berlin, DE", "employment_type": "Full-time"},
    {"id": 12, "name": "Hana Kobayashi", "department": "Data Science", "title": "Data Scientist Intern", "manager": "Priya Ramachandran", "email": "hana.kobayashi@acmecorp.example", "location": "Magdeburg, DE", "employment_type": "Intern"},
]

fieldnames = ["id", "name", "department", "title", "manager", "email", "location", "employment_type"]
with open(EMPLOYEES_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(employees)

print(f"wrote {EMPLOYEES_CSV} ({len(employees)} employees)")
