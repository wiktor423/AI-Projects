% ============================================================
% Variant 2: Number of days between two dates (year 2024)
% Date format: "DDMM" (e.g., "2205" = 22nd of May)
% ============================================================

% --- Number of days in each month (2024 is a leap year) ---
month_length(1,  31).
month_length(2,  29).
month_length(3,  31).
month_length(4,  30).
month_length(5,  31).
month_length(6,  30).
month_length(7,  31).
month_length(8,  31).
month_length(9,  30).
month_length(10, 31).
month_length(11, 30).
month_length(12, 31).

% --- Validate a date: day must be within the month's range ---
valid_date(Day, Month) :-
    month_length(Month, MaxDay),
    Month >= 1, Month =< 12,
    Day   >= 1, Day   =< MaxDay.

% --- Sum of days in all months before month M ---
days_before_month(1, 0).
days_before_month(M, Total) :-
    M > 1,
    M1 is M - 1,
    month_length(M1, L),
    days_before_month(M1, Rest),
    Total is Rest + L.

% --- Parse "DDMM" string and convert to day-of-year ---
% Fails automatically if format is wrong or date is invalid
date_to_day(DateStr, DayOfYear) :-
    atom_chars(DateStr, Chars),
    length(Chars, 4),                        % must be exactly 4 characters
    Chars = [D1, D2, M1, M2],
    number_chars(Day,   [D1, D2]),           % fails if not digits
    number_chars(Month, [M1, M2]),           % fails if not digits
    valid_date(Day, Month),                  % fails if date is out of range
    days_before_month(Month, Offset),
    DayOfYear is Offset + Day.

% --- Main predicate: interval between two dates ---
% Returns the absolute number of days between Date1 and Date2
interval(Date1, Date2) :-
    date_to_day(Date1, Day1),
    date_to_day(Date2, Day2),
    N is abs(Day2 - Day1),                   % abs() handles reversed order
    write(N), nl,
    !.                                        % stop after first solution
