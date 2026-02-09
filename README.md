# Automated-Audit-System-for-Vendor-Invoice-Risk-Analysis
The Automated Audit System for Vendor & Invoice Risk Analysis is a full-stack Python application that automates invoice auditing using OCR-based data extraction, SQL database storage, rule-based verification, and vendor risk analysis.
The system processes invoice images, extracts key fields, verifies financial correctness, calculates vendor and invoice risk scores, and displays insights through a dashboard API.
This project simulates a real-world financial audit automation system used in accounting, compliance, and fraud-prevention workflows.
working link: https://hub.docker.com/repository/docker/praveenanbalagan/auto-audit-system/general

🎯 Core Objectives

Automate invoice data extraction using OCR logic
Store audit results securely using SQL (SQLite)
Verify invoice totals and detect inconsistencies
Calculate vendor and invoice risk scores
Provide dashboard-ready audit analytics
Reduce manual audit effort and errors

⚙️ Key Features
🧾 Invoice OCR Processing

Extracts invoice number, vendor name, totals, and line items
Uses a lightweight OCR simulation engine (PIL-based)
Designed to be easily upgraded to real OCR engines

✅ Invoice Verification Engine

Validates invoice totals against line items
Calculates verification scores
Classifies invoices as:
VERIFIED

NEEDS_REVIEW

FLAGGED

⚠️ Vendor & Invoice Risk Analysis

Rule-based vendor risk profiling
Detects abnormal invoice amounts
Assigns risk levels:

MINIMAL, LOW, MEDIUM, HIGH, CRITICAL

Tracks vendor history and spending behavior

🗄️ SQL Database Integration

Uses SQLite for persistent storage
Stores invoices, audit scores, vendors, and users
Indexed tables for performance
Secure password hashing for authentication

📊 Dashboard Analytics

Total invoices processed
Total spending amount
Average invoice value
Vendor-wise invoice counts and spending
Risk-level distribution across invoices

🔐 Secure API Access

JWT-based authentication
Protected endpoints for invoices and dashboard
Admin login system
<img width="1896" height="932" alt="Screenshot 2026-02-09 202846" src="https://github.com/user-attachments/assets/37b41614-225e-435d-9d16-77589a3081e8" />
<img width="1870" height="901" alt="Screenshot 2026-02-09 202857" src="https://github.com/user-attachments/assets/689ca299-f109-48a6-8fcb-a75693a47c07" />
<img width="1919" height="959" alt="Screenshot 2026-02-09 202911" src="https://github.com/user-attachments/assets/5e9daf90-6c55-45e0-9953-575acc166c79" />
<img width="1538" height="931" alt="Screenshot 2026-02-09 202922" src="https://github.com/user-attachments/assets/1c804623-8a64-4a5b-a224-e01aed3d581f" />



