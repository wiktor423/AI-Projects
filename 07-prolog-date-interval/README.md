# Prolog Date Interval Calculator

A Prolog program to calculate the absolute number of days between two dates in the leap year **2024**.

## Core Features

- **Leap Year Verification**: Configured specifically for the 2024 leap year calendar, where February is correctly defined with 29 days (`month_length(2, 29)`).
- **Date Verification**:
  - Implements the `valid_date/2` validator, checking that months fall within $[1, 12]$ and days are within the boundary specified for the selected month.
- **Dynamic Offset Calculation**:
  - The recursive helper predicate `days_before_month/2` dynamically calculates the cumulative days elapsed in all preceding months.
- **DDMM Input Parsing**:
  - Converts `"DDMM"` formatted string representations (e.g., `"2205"` for May 22nd) into list of atoms, checks length constraint of exactly 4 characters, and parses day and month integer values via `number_chars/2`.
- **Absolute Distance Querying**:
  - The main `interval/2` predicate parses two strings, converts them to day-of-year integers, computes the absolute difference, outputs the result to the console, and prevents backtracking using the cut (`!`) operator.

## Directory Structure

- [main.pl](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/07-prolog-date-interval/main.pl): The complete Prolog program implementing month definitions, validators, parsers, and the main `interval` predicate.
- [Lab7_Report_Variant2.pdf](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/07-prolog-date-interval/Lab7_Report_Variant2.pdf): Final report documenting Prolog predicates, test cases, and variant specification details.

## Setup and Usage

### Online Environment (SWISH)
1. Navigate to [SWISH SWI-Prolog](https://swish.swi-prolog.org).
2. Click on **"Create a program"**.
3. Copy the contents of [main.pl](file:///home/wiktor-klepacz/projects/EITI/EARIN/AI-Labs/07-prolog-date-interval/main.pl) and paste them into SWISH's program editor.
4. Execute queries using the query interface on the bottom right.

### Local Execution (SWI-Prolog CLI)
If you have SWI-Prolog installed locally, you can run the program in your terminal:
```bash
swipl -s main.pl
```
Inside the interactive prompt, execute:
```prolog
?- interval("2205", "0506").
% Output: 14
% true.

?- interval("0102", "1102").
% Output: 10
% true.
```

## Date Format & Constraints
- Dates must be represented as a 4-character string using the `"DDMM"` format.
- Leading zeros are required for single-digit days or months (e.g. use `"0102"` instead of `"102"` or `"12"`).
- Invalid dates (e.g. `"3002"` or `"3201"`) will fail validation and return `false`.

## Authors
- **Savy Timothée**
- **Klepacz Wiktor**
- **Group Number**: 27
