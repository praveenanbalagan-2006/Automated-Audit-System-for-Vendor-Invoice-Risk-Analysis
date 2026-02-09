# Automated-Audit-System-for-Vendor-Invoice-Risk-Analysis
The Automated Audit System for Vendor & Invoice Risk Analysis is a full-stack Python application that automates invoice auditing using OCR-based data extraction, SQL database storage, rule-based verification, and vendor risk analysis.
The system processes invoice images, extracts key fields, verifies financial correctness, calculates vendor and invoice risk scores, and displays insights through a dashboard API.
This project simulates a real-world financial audit automation system used in accounting, compliance, and fraud-prevention workflows.

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
