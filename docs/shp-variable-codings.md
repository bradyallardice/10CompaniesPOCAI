# SHP Variable Codings and Definitions

Source files: Wave 23 (2021) person file and questionnaire as primary; Waves 12, 14, 16, 17, 18, 19 consulted for rotating modules. Long file (`shplong_p_user.dta`) used for harmonized labels.

Variable suffix `$$` = 2-digit year (99, 00, 01, ... 23). Wave 1 = 1999, Wave 25 = 2023.

Standard missing codes apply to all variables unless noted: -8 = other error, -7 = filter error, -3 = inapplicable, -2 = no answer, -1 = does not know.

---

## Labor Market

### WSTAT$$ — Working status (constructed)
**Scale:**
- 1 = active occupied
- 2 = unemployed
- 3 = not in labor force
- -3 = inapplicable, -2 = no answer, -1 = does not know

**Coverage:** All waves (1999–2023)

---

### P$$W01 — Employment: paid work last week
**Question:** "Did you do any paid work last week, even for just a few hours?"
**Scale:** 1 = yes, 2 = no
**Coverage:** All waves (1999–2023)

### P$$W02 — Employment: unpaid work last week
**Question:** "Did you do any unpaid work last week (e.g., in a family business)?"
**Scale:** 1 = yes, 2 = no
**Coverage:** All waves (1999–2023)

### P$$W03 — Has job but not working last week
**Question:** "Do you have a job even though you did not work last week?"
**Scale:** 1 = yes, 2 = no
**Coverage:** All waves (1999–2023)

---

### P$$W18 — Job/employer change: last 12 months
**Question:** "Since [month-year], have you changed your job and/or your employer?"
**Filter:** Employed (W01 or W02 or W03 = 1) and age ≤ 66
**Scale:**
- 1 = yes, only jobs (same employer)
- 2 = yes, employers (same job)
- 3 = yes, jobs and employers
- 4 = no, neither

**Coverage:** All waves (1999–2023)

---

### P$$W29 — Type of employment (constructed)
**Question:** "In your current employment, are you…"
**Filter:** Employed
**Scale:**
- 1 = employed by private household (houseworker, baby-sitter)
- 2 = employee of own public limited or limited liability company
- 3 = self-employed
- 4 = partner in relative's firm
- 5 = employee of another private firm or government organisation

**Coverage:** All waves (1999–2023)

---

### P$$W37 — Contract type
**Question:** "What kind of limited-duration contract do you have?"
**Filter:** P$$W36 = 1 (job is time-limited)
**Scale:**
- 1 = apprenticeship
- 2 = training or voluntary work
- 3 = interim / temporary employment
- 4 = project of limited duration
- 5 = occasional work (e.g. holiday job)
- 6 = seasonal work
- 7 = occupational programme
- 8 = trial period
- 9 = position regularly renewed (e.g. teaching)
- 10 = other

**Coverage:** All waves (1999–2023)

---

### P$$W39 — Full-time vs part-time
**Question:** "Currently, in your main job, do you work part-time or full time (100%)?"
**Filter:** Employed
**Scale:** 1 = part-time, 2 = 100% (full-time)
**Coverage:** All waves (1999–2023)

---

### P$$W74 — Contractual hours per week
**Question:** "How many hours per week are contractually stipulated in your main job?"
**Filter:** P$$W29 = 5 (employee of another firm/government)
**Scale:** Continuous (hours); -5 = number of hours vary
**Coverage:** 2000–2023 (missing 1999)

---

### P$$W77 — Hours worked per week (actual)
**Question:** "How many hours do you usually work per week in your main job, including overtime?"
**Filter:** Employed
**Scale:** Continuous (hours); -5 = number of hours vary
**Coverage:** All waves (1999–2023)

---

### P$$W600 / P$$W601 — Reason for job change (1st / 2nd)
**Question:** "Why did you change profession and/or employer?"
**Filter:** P$$W18 = 1, 2, or 3
**Scale (both variables):**
- 1 = to take up better job (wish for change, advancement, dissatisfaction)
- 2 = end of temporary contract
- 4 = sale/closure of own/family business, end of self-employment
- 5 = child care or care for other dependant
- 6 = partner's job required move / marriage
- 7 = other reasons (seasonal work, journey, timeout, retirement)
- 8 = reduce commuting time
- 9 = change of working time
- 10 = training, internship, studies
- 11 = start self-employment / own business
- 12 = health problems (burnout, accident)
- 13 = conflicts, mobbing, problems with superiors, bad work climate
- 14 = without work before starting actual employment
- 15 = obliged to change: complete closure of company/divisions
- 16 = obliged to change: cuts in manpower, redundancy

**Coverage:** 2004–2023

---

### P$$W602 — Firm restructuring
**Question:** "Has your company undergone a restructuring in the last 12 months?"
**Filter:** P$$W29 = 2, 3, or 5 (employee or self-employed with company)
**Scale:** 1 = yes, 2 = no
**Coverage:** 2004–2023

---

### IS4MAJ$$ — ISCO occupation code, 4-digit (main current job, constructed)
**Scale:** 4-digit ISCO-08 codes; -3 = inapplicable, -2 = no answer; "no corresponding ISCO value" where unmappable
**Coverage:** All waves (1999–2023); requires employment

---

## Income

### I$$EMPMG / I$$EMPMN — Monthly employment income, gross / net
**Scale:** Continuous CHF; -4 = no personal income
**Coverage:** 2002–2023 (missing 1999–2001)

### I$$WYG / I$$WYN — Annual working income, gross / net
**Scale:** Continuous CHF; -5 = irregular, difficult to say; -4 = no personal income
**Coverage:** All waves (1999–2023)

### I$$PTOTG / I$$PTOTN — Total personal income, gross / net
**Scale:** Continuous CHF; -5 = irregular, difficult to say; -4 = no personal income
**Coverage:** All waves (1999–2023)

### P$$I17 — Professional income: extra-month salary / bonuses
**Question:** "As an employee, are you paid a 13th or 14th month salary, a bonus or a gratification?"
**Filter:** Has income from dependent employment
**Scale:** Continuous CHF (amount)
**Coverage:** 1999–2001 only (dropped after 2001)

---

## Job Quality and Working Conditions

### P$$W92 / P$$W93 / P$$W94 / P$$W228 / P$$W229 / P$$W230 — Job satisfaction battery
**Question:** "Can you indicate on a scale from 0 'not at all satisfied' to 10 'completely satisfied' your satisfaction with…"
**Filter:** Employed
**Scale:** 0 (not at all satisfied) to 10 (completely satisfied)

| Variable | Item | Coverage |
|---|---|---|
| P$$W92 | your income | 1999–2023 |
| P$$W93 | your work conditions | 1999–2023 |
| P$$W94 | your work atmosphere | 1999–2023 |
| P$$W228 | your job in general | 1999, then 2004–2023 (gap 2000–2003) |
| P$$W229 | the interest of your tasks | 1999, then 2004–2023 (gap 2000–2003) |
| P$$W230 | the amount of work (workload) | 1999, then 2004–2021 (gap 2000–2003; dropped after 2021) |

---

### P$$W603 — Work intensity / pace
**Question:** "How often do you have to work at a high intensity?"
**Filter:** Employed
**Scale:** 0 (never) to 10 (always)
**Coverage:** 2004–2023

### P$$W604 — Work stress
**Question:** "Does your job expose you to stress?"
**Filter:** Employed
**Scale:** 1 = yes, 2 = no
**Coverage:** 2004–2023

---

### P$$W86 / P$$W86A — Job security
**Question:** "Would you say your job is very secure, quite secure, a bit insecure or very insecure?"
**Filter:** Employed
**Scale:**
- 1 = very secure
- 2 = quite secure
- 3 = a bit insecure
- 4 = very insecure

**Coverage:** `P$$W86A` all waves (1999–2023). Note: no `P$$W86` (without A) in user files — the variable is consistently named W86A. Wording revised at some point; harmonization needed for full-panel use.

---

### P$$W101 — Perceived unemployment risk (next 12 months)
**Question:** "How high do you rate the risk of becoming unemployed in the next 12 months?"
**Filter:** Employed
**Scale:** 0 (no risk at all) to 10 (a sure risk)
**Coverage:** All waves (1999–2023)

---

### P$$W91 — Participation in decisions / autonomy
**Question:** "In the framework of your work, do you participate in decisions?"
**Filter:** Employed
**Scale:**
- 1 = yes, decision (participate in decisions)
- 2 = yes, opinion (give your opinion)
- 3 = no

**Coverage:** All waves (1999–2023)

---

### P$$W71B — Working-time autonomy
**Question:** "Which option best describes the degree of autonomy over your working hours?"
**Filter:** P$$W71A = 4 (flexible working hours)
**Scale:**
- 1 = within some limits, you can decide when to start and end your working day
- 2 = you are completely free to decide how to organise your working day

**Coverage:** 2021–2023 (introduced 2021; only 3 waves)

---

### P$$W80 / P$$W80A — Work from home
**Question:** "How often do you work from home?"
**Filter:** Employed
**Scale (P$$W80A, from ~2017 onward):**
- 1 = in my job it is not possible to work from home
- 2 = never, as I don't want to work from home
- 3 = a few times a year
- 4 = about once a month
- 5 = about once a week
- 6 = several days a week
- 7 = I work mainly at home

**Coverage:** `P$$W80` used 2000–2020 (simpler scale); `P$$W80A` from 2021 onward (different scale — discontinuity at 2021)

---

### P$$W607 — Computer use at work
**Question:** "Do you personally use a computer in your work?"
**Filter:** Employed
**Scale:** 1 = yes, 2 = no
**Coverage:** 2004–2023

---

### P$$W100 — Qualification match
**Question:** "How do you estimate your qualifications with regard to your current job?"
**Filter:** Employed
**Scale:**
- 1 = qualifications are not sufficient (under-qualified)
- 2 = qualifications correspond to the job (matched)
- 3 = qualifications are superior to the job (over-qualified)
- 4 = qualifications do not relate to the job (mismatch)

**Coverage:** All waves (1999–2023)

---

## Political Attitudes

### P$$P10 — Left-right self-placement
**Question:** "In politics, people often talk of 'left' and 'right'. Where would you place yourself personally, if 0 means 'left' and 10 means 'right'?"
**Filter:** age ≥ 16
**Scale:** 0 (left) to 10 (right); -4 = no particular tendency; -5 = cannot situate themselves
**Coverage:** All waves (1999–2023)

---

### P$$P19 — Vote intention
**Question:** "If there was an election for the National Council tomorrow, which party would you vote for?"
**Filter:** age ≥ 16
**Scale (selected):**
- 1 = FDP/PLR (Liberals)
- 2 = CVP/PDC (Christian Democrats)
- 3 = SP/PSS (Social Democrats)
- 4 = SVP/UDC (Swiss People's Party — right-populist)
- 11 = GPS/PES (Greens)
- 20 = GLP/PVL (Green Liberals)
- 21 = BDP (Conservative Democrats)
- 51 = no party
- 52 = wouldn't vote

**Coverage:** All waves (1999–2023)

---

### P$$P72 — Sympathy: Swiss People's Party (SVP/UDC)
**Question:** "Tell me how much you sympathize with the following parties, if 0 means 'not at all' and 10 'completely'… [UDC/SVP]"
**Filter:** age ≥ 16
**Scale:** 0 (not at all) to 10 (completely)
**Coverage:** 2011, 2014, 2017 only (rotating party sympathy module)

---

### P$$P17 — Taxes on high incomes (direction)
**Question:** "Are you in favour of an increase or in favour of a decrease of the tax on high incomes?"
**Filter:** age ≥ 16
**Scale:** 1 = favour increase, 2 = neither, 3 = favour decrease
**Coverage:** 1999–2009 annually; then 2011, 2014, 2017, 2020, 2023 only

---

### P$$P13 — Social spending (direction)
**Question:** "Are you in favour of a diminution or in favour of an increase of the Confederation's social spendings?"
**Filter:** age ≥ 16
**Scale:** 1 = favour increase, 2 = neither, 3 = favour decrease
**Coverage:** 1999–2009 annually; then 2011, 2014, 2017, 2020, 2023 only

---

### P$$P04 — Trust in federal government
**Question:** "How much confidence do you have in the Federal Government (in Bern), if 0 means 'no confidence' and 10 means 'full confidence'?"
**Filter:** age ≥ 16
**Scale:** 0 (no confidence) to 10 (full confidence)
**Coverage:** 1999–2009 annually; then 2011, 2014, 2017, 2020, 2023 only

---

### P$$P45 — General trust in people (social trust)
**Question:** "Would you say that one can trust most people, or that one can't be too careful when dealing with others, if 0 means 'can't be too careful' and 10 means 'most people can be trusted'?"
**Scale:** 0 (can't be too careful) to 10 (most people can be trusted)
**Coverage:** 2002–2023 (missing 1999–2001)

---

### P$$P02 — Satisfaction with democracy
**Question:** "Overall, how satisfied are you with the way democracy works in our country, if 0 means 'not at all satisfied' and 10 'completely satisfied'?"
**Filter:** age ≥ 16
**Scale:** 0 (not at all satisfied) to 10 (completely satisfied)
**Coverage:** 1999–2009 annually; then 2011, 2014, 2017, 2020, 2023 only

---

### P$$P03 — Political efficacy
**Question:** "How much influence do you think someone like you can have on government policy, if 0 means 'no influence' and 10 'a very strong influence'?"
**Filter:** age ≥ 16
**Scale:** 0 (no influence) to 10 (very strong influence)
**Coverage:** 1999–2009 annually; then 2011, 2014, 2017, 2020, 2023 only

---

### P$$P61 — Public spending: unemployment benefits
**Question:** "The government spends money in different sectors. Tell me if you wish the government would spend more, less or the same amount — [Unemployment benefits]"
**Filter:** age ≥ 16
**Scale:** 1 = more, 2 = the same, 3 = less
**Coverage:** 2011, 2014, 2017 only

### P$$P63 — Public spending: social aid
**Question:** Same spending battery — "Social aid (for the poor)"
**Scale:** 1 = more, 2 = the same, 3 = less
**Coverage:** 2011, 2014, 2017 only

---

### P$$P14 — Opinion on EU accession
**Question:** "Are you in favour of Switzerland joining the European Union or are you in favour of Switzerland staying outside of the European Union?"
**Filter:** age ≥ 16
**Scale:** 1 = favour joining the EU, 2 = neither, 3 = favour staying outside the EU
**Coverage:** 1999–2009 annually; then 2011, 2014, 2017, 2020, 2023 only

---

### P$$P15 — Opportunities for foreigners
**Question:** "Are you in favour of Switzerland offering foreigners the same opportunities as Swiss citizens, or in favour of offering Swiss citizens better opportunities?"
**Filter:** age ≥ 16
**Scale:** 1 = favour equality of opportunities, 2 = neither, 3 = favour better opportunities for Swiss citizens
**Coverage:** 1999–2009 annually; then 2011, 2014, 2017, 2020, 2023 only

---

### P$$P87–P$$P93 — Anomie scale (7 items)
**Introduction:** "Please tell me how far you would agree with the following statements, if 0 means 'I completely disagree' and 10 'I completely agree'."
**Filter:** age ≥ 16; rotating module only (not every wave)
**Scale:** 0 (completely disagree) to 10 (completely agree)
**Coverage:** 2014, 2018, 2021 only

| Variable | Statement |
|---|---|
| P$$P87 | "With everything so uncertain these days, it almost seems as though anything could happen." |
| P$$P88 | "What is lacking in the world today is the old kind of friendship that lasted for a lifetime." |
| P$$P89 | "With everything in such a state of disorder, it's hard for a person to know where he stands from one day to the next." |
| P$$P90 | "Everything changes so quickly these days that I often have trouble deciding which are the right rules to follow." |
| P$$P91 | "I often feel that many things that our parents stood for are just going to ruin before our very eyes." |
| P$$P92 | "The trouble with the world today is that most people really don't believe in anything." |
| P$$P93 | "People were better off in the old days when everyone knew just how he was expected to act." |

---

## Subjective Well-being and Mental Health

### P$$C44 — Life satisfaction
**Question:** "In general, how satisfied are you with your life if 0 means 'not at all satisfied' and 10 means 'completely satisfied'?"
**Scale:** 0 (not at all satisfied) to 10 (completely satisfied)
**Coverage:** 2000–2023 (missing 1999 only)

---

### P$$C17 — Depression, blues, anxiety: frequency
**Question:** "Do you often have negative feelings such as having the blues, being desperate, suffering from anxiety or depression, if 0 means 'never' and 10 'always'?"
**Scale:** 0 (never) to 10 (always)
**Coverage:** All waves (1999–2023)

---

### P$$C205 / P$$C206 / P$$C207 — Illness diagnoses
**Question:** "Have you had any of the following health problems, either since [month/year] or chronically? [A doctor has diagnosed this.]"
**Filter:** Respondent has reported illness or relevant life event
**Scale:** 1 = yes, 2 = no

| Variable | Illness | Coverage |
|---|---|---|
| P$$C205 | Burnout | 2021–2023 |
| P$$C206 | Depression | 2021–2023 |
| P$$C207 | Other psychological problem | 2021–2023 |

---

### P$$C180–P$$C184 — Perceived Stress Scale (PSS)
**Introduction:** "Here is a series of questions about your feelings and thoughts during the LAST MONTH. Please indicate how often you felt or thought a certain way."
**Scale:** 1 = never, 2 = almost never, 3 = sometimes, 4 = fairly often, 5 = very often

| Variable | Item | Coverage |
|---|---|---|
| P$$C180 | Felt able to control important things in your life | 2016 only |
| P$$C181 | Felt unable to control important things | 2016 only |
| P$$C182 | Felt that things were going your way | 2016 only |
| P$$C183 | Felt difficulties piling up so high you could not overcome them | 2016 only |
| P$$C184 | Felt stressed / nervous during the last month | 2016–2023 |

Note: C180–C183 were a one-time 2016 module. Only C184 was retained thereafter.

---

### P$$C112–P$$C114 — Financial worries
**Introduction:** "Tell me how much you worry about the following topics, where 0 means 'no worry at all' and 10 'a great deal of worry'."
**Scale:** 0 (no worry at all) to 10 (a great deal of worry)
**Coverage:** 2012 only (one-time module)

| Variable | Item |
|---|---|
| P$$C112 | Not being able to afford things |
| P$$C113 | Feeling insecure |
| P$$C114 | Not being able to pay bills |

---

### P$$C70–P$$C74 — Locus of control (5 items)
**Introduction:** "Please tell me how well do the following statements describe your personality, if 0 means 'I completely disagree' and 10 'I completely agree'."
**Scale:** 0 (completely disagree) to 10 (completely agree)
**Coverage:** 2009, 2012, 2015, 2018, 2021 (every 3 years)

| Variable | Statement |
|---|---|
| P$$C70 | "Often it is not worth making plans, because too much is unpredictable." |
| P$$C71 | "I feel like I have little influence on the events of my life." |
| P$$C72 | "I am easily able to overcome unexpected problems." |
| P$$C73 | "In general, I have no difficulty choosing between two possibilities." |
| P$$C74 | "Sometimes I feel useless." |

---

## Education and Training

### P$$E14 — Currently enrolled in training / school
**Question:** "Are you currently studying at a school? [Does not include training courses.]"
**Filter:** age < 67
**Scale:** 1 = yes, 2 = no
**Coverage:** All waves (1999–2023)

### P$$E15 — Type of current training (constructed)
**Question:** "What type of training is it?"
**Filter:** age < 67
**Scale (selected categories):**
- 0 = incomplete compulsory school
- 1 = only completed compulsory school
- 3 = apprenticeship (CFC/EFZ)
- 7 = bachelor/maturity (high school)
- 15 = university / EPF / ETH (bachelor, master, doctorate)
- 17 = university of applied sciences (HES/FH)

**Coverage:** All waves (1999–2023)

### P$$E18 — Professional training courses: last 12 months
**Question:** "Since [month-year], have you attended one or several professional training courses for professional reasons?"
**Filter:** age < 67; excludes those currently in full-time education (P$$E14 = 1)
**Scale:** 1 = yes, one, 2 = yes, several, 3 = no
**Coverage:** All waves (1999–2023)

### EDCAT$$ — Highest level of education (17 categories, constructed)
**Scale (main categories):** incomplete compulsory school / compulsory school / apprenticeship (CFC/EFZ) / full-time vocational school / vocational maturity / general training school / bachelor/maturity / technical or vocational school / university of applied sciences (HES/FH) / university/academic high school/EPF/ETH / PhD
**Coverage:** All waves (1999–2023)

### ISCED$$ — Education level (ISCED 1997, constructed)
**Scale:**
- 0 = not completed primary education
- 10 = primary education
- 20 = lower secondary
- 31/32/33 = upper secondary (tracks A/B/C)
- 41 = post-secondary non-tertiary
- 51 = tertiary 5A (general)
- 52 = tertiary 5B (professional)
- 61 = second stage of tertiary (PhD)

**Coverage:** All waves (1999–2023)

---

## Firm and Industry Characteristics

### NOGA2M$$ — Industry classification (constructed)
**Type:** Constructed by SHP staff from open-text employer activity; coded to Swiss NOGA (Nomenclature générale des activités économiques, equivalent to NACE Rev. 1), then recoded to 17 broad sectors. The underlying 5-digit NOGA1M code is not released in user files.
**Filter:** Employed (W01 or W02 or W03 = 1)
**Coverage:** All waves (1999–2023); ~137k valid person-year obs in long file

**Scale (17 categories):**

| Code | Sector |
|---|---|
| 1 | Agriculture, hunting, forestry |
| 2 | Fishing and fish farming |
| 3 | Mining and quarrying |
| 4 | Manufacturing |
| 5 | Electricity, gas and water supply |
| 6 | Construction |
| 7 | Wholesale/retail; repair of motor vehicles and household goods |
| 8 | Hotels and restaurants |
| 9 | Transport, storage and communication |
| 10 | Financial intermediation / insurance |
| 11 | Real estate; renting; computer; research |
| 12 | Public administration / national defence / compulsory social security |
| 13 | Education |
| 14 | Health and social work |
| 15 | Other community, social and personal service activities |
| 16 | Private households with employed persons |
| 17 | Extra-territorial organizations and bodies |

Note: The 5-digit NOGA1M code (finer classification) is constructed internally by SHP but not released in user files. Request from SHP directly if sub-sector detail is needed.

---

### P$$W32 — Private vs public employer
**Question:** "Are you employed by a private company or a state organisation?"
**Filter:** Employees only (P$$W29 = 5); excludes self-employed
**Scale:** 1 = private company, 2 = government organisation
**Coverage:** All waves (1999–2023); ~111k valid obs in long file

---

### P$$W33 — Type of public employer (level of government)
**Question:** Follow-up to W32 for those employed by government
**Filter:** P$$W32 = 2 (government)
**Scale:**
- 1 = international organisation
- 2 = Confederation / Swiss Railways / Post office
- 3 = Canton
- 4 = Commune

**Coverage:** 1999–2003 only (dropped after Wave 5)

---

### P$$W85 — Firm size (number of employees, company level)
**Question:** "How many persons are employed in your company (association/institution)?"
**Filter:** Employees only (P$$W29 = 5)
**Scale:**
- 1 = 1 to 4
- 2 = 5 to 9
- 3 = 10 to 19
- 4 = 20 to 24
- 5 = 25 to 49
- 6 = 50 to 99
- 7 = 100 to 499
- 8 = 500 to 999
- 9 = over 1,000

**Coverage:** All waves (1999–2023); ~110k valid obs in long file

---

### P$$W31 — Firm size (self-employed: number of employees, excluding self)
**Question:** "How many persons do you employ on a regular basis, not counting yourself?"
**Filter:** Self-employed only (P$$W29 = 2 or 3)
**Scale:** 1 = 0, 2 = 1–4, 3 = 5–9, 4 = 10–24, 5 = 25–49, 6 = 50–99, 7 = 100 and over
**Coverage:** All waves (1999–2023); ~21k valid obs

---

## Summary: Coverage by Category

| Category | Mostly full coverage | Rotating / sparse |
|---|---|---|
| Labor market | WSTAT, W01–03, W18, W29, W37, W39, W74, W77, W86A, W91, W100, W101, IS4MAJ | W600/601 (from 2004), W602 (from 2004), W71B (from 2021) |
| Income | WYG/WYN, PTOTG/PTOTN | EMPMG/MN (from 2002), I17 (1999–2001 only) |
| Job quality | W92/93/94, W101 | W228/229/230 (gap 2000–2003), W603/604/607 (from 2004), W80A (from 2021) |
| Politics | P10, P19 | P02/03/04/13/14/15/17 (annual to 2009, then every 3 yrs), P45 (from 2002), P72/61/63 (3 waves only), P87–P93 (3 waves only) |
| Well-being | C44 (from 2000), C17 | C70–74 (every 3 yrs), C184 (from 2016), C180–183 (2016 only), C112–114 (2012 only), C205–207 (from 2021) |
| Training | E14, E15, E18, EDCAT, ISCED | — |
| Firm/industry | NOGA2M (industry, 17 cats), W32 (private/public), W85 (firm size) | W33 (type of public employer, 1999–2003 only) |
