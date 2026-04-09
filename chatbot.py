"""
╔══════════════════════════════════════════════════════════════╗
║           ATHENURA CHATBOT  —  AI/ML Enhanced               ║
║    TF-IDF + Cosine Similarity  |  Keyword Rules  |  NLP     ║
╚══════════════════════════════════════════════════════════════╝

Run:
    python chatbot.py

No external model files needed — fully self-contained.
"""

import re
import datetime
import math
from collections import defaultdict
from typing import Optional


# ══════════════════════════════════════════════════════════════
#  SECTION 1 — Q&A KNOWLEDGE BASE
# ══════════════════════════════════════════════════════════════

QA_DICT: dict[str, str] = {
    # ── Company Overview ──────────────────────────────────────
    "what is athenura": "Athenura is an EdTech and IT services organization focused on bridging academic learning with industry requirements.",
    "what services does athenura provide": "Athenura provides internships, training programs, mentorship, and live project exposure.",
    "is athenura reliable": "Yes, Athenura is a trusted EdTech and IT services platform with structured internship programs.",
    "what is the office address of athenura": "Athenura is located at Sector 62, Block B Industrial Area, Noida, Uttar Pradesh, India.",
    "is athenura located in noida": "Yes, Athenura operates from Noida Sector 62, Block B Industrial Area, Uttar Pradesh, India.",
    "which city is athenura based in": "Athenura is based in Noida, Uttar Pradesh, India.",
    "what is the exact location of athenura": "Athenura is located at Sector 62, Block B Industrial Area, Noida, Uttar Pradesh, India.",
    "is athenura located in an industrial area": "Yes, Athenura operates from Block B Industrial Area, Sector 62, Noida.",
    "how can i visit the athenura office": "You can visit the office at Sector 62, Block B Industrial Area, Noida. It is recommended to schedule your visit in advance.",
    "what landmark is near athenura": "Athenura is located in the industrial hub of Sector 62, Noida, near major corporate offices and metro connectivity.",
    "is athenura accessible from delhi": "Yes, Athenura in Noida Sector 62 is easily accessible from Delhi via road and metro connectivity.",
    "does athenura operate only from noida": "The main office is located in Noida Sector 62, but Athenura offers services remotely to clients and interns globally.",
    "is athenura a registered company": "Yes. Athenura operates under a legally registered entity with formal corporate structure.",
    "is athenura a private limited company": "Yes. Athenura operates with a formal corporate structure including directors and defined governance.",
    "what is the legal name of the company": "The official legal name of the company is Athenura.",
    "what is athenura official website": "Athenura's official website is https://www.athenura.in where you can apply for internships, access portals, and view company information.",
    "does athenura have social media pages": "Yes. Athenura maintains official pages on LinkedIn, Instagram, Facebook, and other platforms for updates and announcements.",
    "where can i see athenura public work or updates": "Follow Athenura on LinkedIn, Instagram, Facebook, X, Reddit, YouTube etc for updates, tutorials, and examples of work.",
    "what is the company hierarchy at athenura": "The hierarchy is Founder or Directors, followed by Senior Management, Head of Departments, Team Leaders, and then Interns.",
    "what is the reporting structure in athenura": "Interns report to Team Leaders, who report to Department Heads, who report to senior management.",
    "who supervises interns": "Interns are primarily supervised by their Team Leaders.",
    "who monitors intern performance": "Performance is monitored by Team Leaders, Department Heads, and the HR department.",
    "who approves internship certificates": "Certificates are approved by HR and company management after verification.",

    # ── Contact Information ────────────────────────────────────
    "what is the official hr contact email": "HR: hr.athenura@gmail.com. Official company email: official@athenura.in.",
    "what is the hr email id": "The official HR email is hr.athenura@gmail.com. It can be used for certificates, internship verification, resignation requests, and HR queries.",
    "what is the hr contact number": "HR contact numbers are usually shared during onboarding. For official matters email communication is preferred.",
    "what is the contact number of athenura": "You can contact Athenura at +91 98350 51934.",
    "what phone number can i use to contact athenura": "The official contact number is +91 98350 51934.",
    "how can i contact athenura for support": "You can call +91 98350 51934 for support or email hr.athenura@gmail.com for HR-related queries.",
    "how can clients contact athenura": "Clients can contact Athenura by calling +91 98350 51934 or emailing official@athenura.in.",
    "what is the official support number for athenura": "The official support number is +91 98350 51934.",
    "can i call athenura for business inquiries": "Yes, business inquiries can be made at +91 98350 51934 or via email at official@athenura.in.",
    "is there a phone number for client support at athenura": "Yes, client support is available at +91 98350 51934.",
    "how do i reach athenura quickly": "The fastest way is to call +91 98350 51934.",
    "what number should companies call to partner with athenura": "Companies can call +91 98350 51934 to discuss partnership opportunities.",
    "can i contact athenura by phone": "Yes, you can call +91 98350 51934 to contact Athenura.",
    "how can i contact hr urgently": "First contact your Team Leader and then send an email to HR explaining the issue clearly.",
    "what are hr working hours": "HR working hours are typically Monday to Saturday from 10:00 AM to 6:00 PM IST.",
    "who should i contact for certificate issues": "Certificate issues should be reported to HR with your intern name, intern ID, issue description, and screenshot if necessary.",
    "who should i contact for technical issues": "Technical issues should be reported to the technical support team or the HR department.",
    "where should i send my resignation email": "Resignation emails should be sent to the official HR email address: hr.athenura@gmail.com.",
    "how do i escalate an issue": "The escalation hierarchy is Intern → Team Leader → HR → Management.",
    "what is the official email for athenura": "The official company email is official@athenura.in and HR can be reached at hr.athenura@gmail.com.",

    # ── Internship Overview ───────────────────────────────────
    "what is the objective of the internship": "The objective is to provide practical industry exposure and skill development.",
    "is the internship practical based": "Yes, the internship is focused on hands-on learning and live projects.",
    "is internship beneficial": "Yes, it is highly beneficial for skill development and career growth.",
    "is internship worth it": "Yes, the internship adds real industry experience to your resume and builds practical skills.",
    "is internship flexible": "Yes, Athenura internships are flexible and student-friendly.",
    "is internship interactive": "Yes, the program includes interactive sessions, team collaboration, and live projects.",
    "is the internship online": "Yes, the internship is fully remote and conducted online.",
    "is the program structured": "Yes, the internship follows a structured schedule with clear milestones.",
    "is it beginner friendly": "Yes, no prior experience is required. The program is suitable for beginners.",

    # ── Eligibility & Application ─────────────────────────────
    "who can apply for internships": "Students, graduates, and early-career professionals are eligible to apply.",
    "how can i apply": "You can apply through the official Athenura website at https://www.athenura.in/apply-internship or partner institutions.",
    "how do i enroll in internship": "You can enroll through the official website at www.athenura.in.",
    "is enrollment confirmation immediate": "Yes, you will receive a confirmation after successful registration.",
    "will i receive login details": "Yes, login credentials are sent to your registered email after enrollment.",
    "can i enroll anytime": "Yes, enrollment is open year-round.",
    "is there limited seats": "Yes, seats may be limited per batch. Apply early to secure your spot.",
    "application review": "Application review is managed under company supervision and follows structured evaluation processes.",
    "how does athenura handle selection criteria": "Selection criteria is managed under company supervision with structured and transparent processes.",
    "where can i apply for an internship at athenura": "You can apply for internships directly through the official website: https://www.athenura.in/apply-internship",
    "where can i find official internship information": "Official internship details are available on www.athenura.in, and on the company's LinkedIn and Instagram pages.",
    "does athenura post internship openings externally": "Yes — openings are posted on the official website, LinkedIn, and internship portals where Athenura lists roles.",
    "what internship domains are available at athenura": "Athenura offers internships across multiple domains including Digital Marketing, Graphic Design, Video Editing, UI/UX, Content Writing, Frontend, Backend, MERN/Full-Stack, Data Science & Analytics, Social Media Management, Sales & Marketing, Email & Outreach, and Human Resources.",
    "will interns work on real or live projects": "Yes — interns are assigned to live projects for real-world exposure and practical skill development.",
    "what documents are required for onboarding": "Typically CV, college ID, and signed internship agreement (if applicable).",
    "what should i do if i face onboarding issues": "Contact your assigned Team Head first. If unresolved, escalate to HR at hr.athenura@gmail.com.",
    "is athenura internship valid for college submission": "Many colleges accept internships from registered organizations. However final acceptance depends on the rules of the specific college or university.",
    "how can i verify an athenura internship": "Internship verification can be done using the intern unique ID, offer letter, official certificate, HR email verification, or the internship portal verification system.",

    # ── Interview ─────────────────────────────────────────────
    "when will my interview be scheduled": "Interview schedules are shared after the application screening process through email, HR communication, or portal notification.",
    "how will i receive the interview link": "The interview link is usually sent through your registered email address or official internship communication groups.",
    "what is the interview format": "Most interviews are conducted online and may include introduction, academic discussion, skill evaluation, and domain related questions.",
    "is the interview hr based or technical": "It depends on the role. HR roles usually have HR interviews, technical roles have technical interviews, and marketing roles may have skill based interviews.",
    "can i reschedule the interview": "Yes. Rescheduling is possible in genuine cases such as exams or medical emergencies. Inform HR in advance.",
    "what happens if i miss the interview": "Missing the interview without informing HR may be treated as non participation. However you may request rescheduling if you have a valid reason.",
    "when will interview results be announced": "Results are usually announced after the evaluation process is completed depending on the number of applicants and internal review.",
    "will i receive feedback after the interview": "Individual feedback may not always be provided but selected candidates will receive confirmation from HR.",

    # ── Induction ─────────────────────────────────────────────
    "when is the internship induction conducted": "Induction is conducted at the beginning of the internship, usually within the first few days after interns receive their offer letter and onboarding instructions.",
    "is the induction session compulsory": "Yes. Induction is mandatory because it explains internship policies, task workflows, communication rules, and evaluation criteria.",
    "what should i do if i miss the induction session": "Inform your Team Leader or HR immediately. You may be asked to watch the recording or attend the next session.",
    "are induction sessions recorded": "Yes. Induction sessions are usually recorded and may be shared for interns who faced technical issues or unavoidable circumstances.",
    "how will i receive the induction meeting link": "The link is shared through official communication channels such as email, intern communication groups, or the internship portal dashboard.",
    "is attendance recorded during induction": "Yes. Attendance is usually recorded during induction sessions and interns must join with their registered name.",
    "what topics are covered in the induction session": "Topics include company introduction, internship structure, reporting hierarchy, task process, attendance rules, performance evaluation, certification criteria, and communication channels.",

    # ── Duration ──────────────────────────────────────────────
    "what is the internship duration": "Internships commonly run 3–8 months depending on the role. Specific duration will be mentioned in your offer details.",
    "can i extend internship": "Yes, extension may be allowed in special cases.",
    "can i complete early": "Yes, early completion is allowed if all tasks are finished.",

    # ── Mode & Schedule ───────────────────────────────────────
    "is hybrid mode available": "Hybrid mode is available in some cases.",
    "is offline internship possible": "No, the internship is entirely online.",
    "do i need to visit the office": "No office visit is required.",
    "is flexible timing allowed": "Yes, flexible timing is supported.",
    "can i choose my schedule": "Yes, flexible scheduling is allowed.",
    "are deadlines strict": "Yes, deadlines must be followed to ensure successful completion.",
    "will i get a timetable": "Yes, a schedule is provided at the start of the internship.",
    "are sessions daily": "Session frequency depends on the program structure.",
    "can i manage with college": "Yes, the internship is student-friendly and can be balanced with college.",
    "can i manage with studies": "Yes, the flexible schedule helps you balance internship with studies.",
    "what is the working schedule": "The internship follows a 5-day working culture with active participation expected on working days.",
    "what is the weekly schedule": "Interns follow a 5-day working schedule with tasks, collaboration, and project submissions.",
    "are daily meetings conducted": "No — daily meetings are not mandatory. Meetings are scheduled in the Planner according to departments and project needs.",
    "what happens if i am late to a meeting": "Professional punctuality is expected. Repeated delays may lead to warnings as per disciplinary guidelines.",
    "how do i record my attendance": "Follow the attendance process communicated by your Team Head (could be via official group check-in or a time-tracking tool).",

    # ── Stipend ───────────────────────────────────────────────
    "is the internship paid": "The internship is primarily unpaid and focused on learning. Some domains or project-based contributions may have stipend provisions based on performance.",
    "will i earn a stipend": "No guaranteed stipend. Some project contributions may have stipend provisions based on performance.",
    "what is the stipend": "The internship is primarily unpaid. The focus is on practical learning and career development.",
    "which interns are eligible for stipends": "Some domains or specific project contributions may have stipend provisions. Stipends are performance/project-based and not guaranteed.",
    "how are project sales shared with interns": "If an intern contributes to a client sale, a percentage may be shared among contributors — the split is decided based on contribution and Team Head agreement.",
    "when and how are stipends or incentives paid": "Payment methods and schedules (if applicable) are communicated by HR/Accounts per project. Always request written confirmation for any stipend-related agreement.",
    "are reimbursements available for internship costs": "Routine reimbursements aren't standard for virtual internships; if any reimbursement is applicable it will be communicated per project and pre-approved.",

    # ── Certificate ───────────────────────────────────────────
    "will i receive a certificate": "Yes, you will receive an Internship Completion Certificate (ICC) upon successful completion and meeting attendance/performance criteria.",
    "will certificate mention duration": "Yes, the certificate includes your internship duration.",
    "is certificate signed": "Yes, the certificate carries an authorized signature.",
    "can i verify certificate online": "Yes, certificate verification is available online through the intern portal.",
    "will certificate include skills": "Yes, the skills you worked on are mentioned in the certificate.",
    "is certificate printable": "Yes, the certificate is available in a printable format.",
    "are certificates graded": "Yes, certificates may reflect your performance level.",
    "certificate eligibility": "Eligibility requires internship completion, task submissions, and meeting the minimum performance score.",
    "is completion mandatory for certificate": "Yes, completion of all tasks is mandatory to receive the certificate.",
    "when and how will i receive my internship certificate": "Certificates (ICC) are issued after successful completion and meeting attendance/performance criteria. Delivery method (digital/PDF or hardcopy) will be communicated by HR.",
    "how can i download my certificate": "Certificates can be downloaded from the intern portal certificate section once you have completed the internship and met the criteria.",
    "what are the certificate eligibility criteria": "Eligibility requires internship completion, task submissions, and meeting the minimum performance score.",
    "can certificate details be corrected": "Yes. Contact HR with the correct information to request a correction.",
    "how long after internship end do i get my lor or certificate": "Typically after exit formalities are complete and performance checks are done. Confirm exact timelines with HR during exit.",

    # ── LOR ───────────────────────────────────────────────────
    "lor issuance": "LOR issuance guidelines are shared during induction and training.",
    "how does athenura handle lor issuance": "LOR issuance follows Athenura's official internship policies shared during induction.",
    "what should interns know about lor issuance": "LOR issuance is structured according to Athenura's official policies. Guidelines are shared during training.",
    "is lor issuance mandatory": "LOR issuance is subject to performance and completion of the internship.",
    "how can i get a letter of recommendation lor": "LORs are awarded to outstanding performers; demonstrate strong performance, reliability, and initiative to be considered.",
    "can i receive a letter of recommendation": "Yes. Letters of Recommendation are usually given to high performing interns with excellent evaluation and participation.",
    "what is lor": "LOR stands for Letter of Recommendation, which is a document that vouches for an individual's skills, character, and performance.",
    "what are the rules for lor issuance": "Guidelines about LOR issuance are shared during induction and training.",
    "will i get references after leaving": "Good performers may receive Letters of Recommendation or references from Team Heads — request these before exit and ensure your work record is clean.",

    # ── Placement ─────────────────────────────────────────────
    "is placement guaranteed": "No, placement is not guaranteed. However, opportunities are shared based on performance.",
    "will i get job opportunities": "Yes, job opportunities are shared with interns based on their performance and skills.",
    "can i join full time": "Exceptional interns may receive placement offers at Athenura. Every candidate is provided a placement opportunity based on performance.",
    "placement support": "Athenura ensures professional standards in all matters related to placement support.",
    "when does placement take place": "Placement opportunities are generally conducted after internship completion and announced by HR.",
    "is placement guaranteed after the internship": "No. Placement depends on performance evaluation, company requirements, and interview results.",
    "how do i register for placement": "Eligible interns must complete the placement registration form shared by HR.",
    "what are the placement eligibility criteria": "Criteria usually include internship completion, task submissions, good performance score, and attendance compliance.",
    "is placement performance based": "Yes. Placement decisions depend significantly on intern performance evaluation.",
    "how many rounds are there in placement": "Typical placement includes resume screening, interview round, and final evaluation.",
    "will hr inform shortlisted candidates": "Yes. HR informs shortlisted candidates through email or official communication channels.",
    "how are placement or pre-placement offers handled": "Exceptional interns may receive placement offers at Athenura; average performers may be referred to partner HR firms. Everyone gets placement opportunity.",

    # ── PPO ───────────────────────────────────────────────────
    "ppo opportunities": "PPO (Pre-Placement Offer) opportunities are available and shared with top-performing interns.",
    "how does athenura handle ppo opportunities": "PPO opportunities are shared with interns based on evaluation and performance criteria.",
    "what is a ppo": "A Pre Placement Offer (PPO) is a job offer given to top performing interns before the formal placement process.",
    "how important is ppo opportunities at athenura": "PPO opportunities are managed under company supervision and structured processes.",
    "do interns receive guidance on ppo opportunities": "Interns are expected to follow all rules related to PPO opportunities.",

    # ── Onboarding ────────────────────────────────────────────
    "will there be an onboarding session": "Yes, an onboarding session is conducted at the start of the internship.",
    "what happens in onboarding": "Onboarding includes an introduction to the program, tools, team, and guidelines.",
    "is onboarding mandatory": "Yes, onboarding is mandatory and important for a smooth start.",
    "will i meet mentors during onboarding": "Yes, you will be introduced to your mentors during the onboarding session.",
    "are guidelines explained during onboarding": "Yes, all guidelines are clearly explained during onboarding.",

    # ── Learning & Training ───────────────────────────────────
    "is prior experience required": "No prior experience is required. The internship is beginner-friendly.",
    "will i learn industry practices": "Yes, you will learn current industry practices through real projects.",
    "are case studies included": "Yes, real-world case studies are included in the program.",
    "is peer learning encouraged": "Yes, collaboration and peer learning are actively encouraged.",
    "will i get study materials": "Yes, study materials and resources are provided throughout the internship.",
    "is learning practical": "Yes, the learning approach is hands-on and practical.",
    "internship training": "Internship training is a key component of the evaluation and development process.",
    "will i learn new things": "Yes, continuous learning is a core part of the Athenura internship.",
    "are there workshops or guest lectures": "Yes — Athenura schedules sessions, workshops, and guest talks as part of the club and learning activities.",
    "what is the saturday club meeting": "A cross-domain session with guest speakers from diverse fields sharing experiences; debates and fun sessions are conducted. Attendance is expected.",
    "where can i find help articles or tutorials from athenura": "Follow Athenura's LinkedIn, Instagram, and website blog pages — they post creative tutorials and tips periodically.",

    # ── Technology ────────────────────────────────────────────
    "will i learn latest technologies": "Yes, the internship covers updated and industry-relevant technologies.",
    "are tools industry standard": "Yes, only industry-standard tools and technologies are used.",
    "do i need to install software": "Yes, required tools must be installed. Setup guidance is provided.",
    "is technical setup guided": "Yes, technical setup assistance is provided to all interns.",
    "will i get technical resources": "Yes, technical resources and references are shared.",
    "do i need coding skills": "Basic coding knowledge is helpful but not always mandatory depending on your role.",
    "will i build projects": "Yes, project-based learning is a central part of the internship.",
    "is git used": "Yes, version control tools like Git are used during the internship.",
    "is debugging taught": "Yes, debugging and problem-solving are included in the practical training.",
    "what tools will i use during the internship": "Tools vary by domain (e.g., Adobe/Canva for design, code editors for developers, analytics tools for marketing). Team Heads will provide access and training.",
    "will athenura provide software licenses or accounts": "The company provides access to required company tools/accounts for project work. Personal licenses may not be provided — check with your Team Head.",
    "are there training resources available": "Yes — interns get hands-on tool exposure, workshops, and guidance during the program; check internal resources shared by your Team Head.",
    "as a design intern what tools should i be familiar with": "Familiarity with Canva and Adobe Creative Cloud (Photoshop, Illustrator, InDesign) is preferred; Team Heads may assign tool-specific guidance.",
    "as a developer intern what stack will i use": "Depending on placement — frontend, backend, MERN/full-stack — expect tools like React, Node, Express, MongoDB, and code editors. Team Heads provide project specifics.",
    "as a marketing intern what metrics will i work with": "Campaign KPIs, reach, engagement, CTR, conversions, SEO metrics, and analytics dashboards — you'll gain hands-on exposure.",
    "what if i lose access to an official account or tool": "Report immediately to your Team Head and IT/HR so access can be restored. Do not try to access accounts beyond your permission.",
    "can i request change of domain mid-internship": "Domain changes are not encouraged. If you want to do it, inform your Team Head and you will be notified about the procedure. In most cases, you may need to appear for the interview again.",
    "will athenura run background or reference checks": "If converting to employment, standard pre-employment/reference checks may apply. For internships, ID/college verification is typical.",

    # ── Teamwork ──────────────────────────────────────────────
    "will i work in teams": "Yes, team collaboration is a key component of the internship.",
    "are team leads assigned": "Yes, team leaders are assigned to guide and coordinate work.",
    "is communication important": "Yes, effective communication is essential during the internship.",
    "are group discussions held": "Yes, group discussions and meetings are conducted regularly.",
    "can i lead a team": "Yes, high-performing interns may get the opportunity to lead a team.",
    "are communication tools used": "Yes, tools like WhatsApp, email, Google Meet, and official communication channels are used.",
    "team collaboration": "Team collaboration is an important component of the internship framework.",
    "who do i report to daily": "Report to your Team Head for daily tasks. Hierarchy: Intern → Team Head → Team Leader → Manager. Escalate to HR only if unresolved.",
    "can i message seniors privately outside official channels": "No — all official communication should happen within assigned official groups/email. Private messaging without permission is discouraged.",
    "what if there is a grievance with my team head": "Follow hierarchy: raise the issue to the next level. If unresolved, escalate to HR at hr.athenura@gmail.com.",
    "how do i file a grievance": "Raise it first with your senior or Team Head. If unresolved, escalate to HR (hr.athenura@gmail.com). Avoid informal or gossip channels.",
    "what communication channels do you use": "We communicate via WhatsApp, email, Google Meet, and official communication channels.",

    # ── Attendance ────────────────────────────────────────────
    "is attendance compulsory": "Yes, a minimum attendance percentage is required. All interns must comply with attendance policies.",
    "how is attendance tracked": "Attendance is tracked through the online intern portal and official check-in processes communicated by your Team Head.",
    "can i take leave": "Yes, leave can be taken with prior permission. Apply 48 hours in advance with documents via the official leave portal.",
    "will absence affect performance": "Yes, excessive absences may negatively impact your evaluation.",
    "is attendance flexible": "Attendance has some flexibility, but the minimum requirement must be met.",
    "how many holidays are provided": "Interns are provided with 4 official holidays during the internship period.",
    "are emergency leaves available": "Yes, interns are eligible for 2 emergency leaves during the internship.",
    "how can interns apply for emergency leave": "Interns can apply for emergency leave at: www.athenura.in/internship/leave",
    "where can i apply for leave": "Apply for leaves from https://www.athenura.in/internship/leave",
    "how many leaves are allowed during the internship": "3-month internship: 10 days | 4-month internship: 12 days | 6-month internship: 20 days | Emergency leaves: 2 additional days.",
    "how do i apply for leave": "Apply 48 hours in advance with documents via the official leave portal. Emergency leave requires immediate intimation to your senior.",
    "what is considered unauthorized leave": "3 consecutive days without approval is treated as unauthorized and may lead to termination and college notification.",
    "what is the policy for sick or emergency leave": "Inform your senior immediately and submit documents if requested. Emergency leave becomes valid after acknowledgment.",
    "what happens if i exceed my leave quota": "Exceeding quota may affect certificate eligibility or require internship extension as per company policy.",

    # ── Mentorship ────────────────────────────────────────────
    "will mentors guide projects": "Yes, mentors provide regular guidance on projects.",
    "can i contact mentors anytime": "Mentors are available during working hours for queries.",
    "are one on one sessions available": "Yes, one-on-one sessions are available if needed.",
    "do mentors review work": "Yes, mentors review your work and provide constructive feedback.",
    "is career guidance included": "Yes, career advice and guidance are provided throughout the program.",
    "how often are mentor sessions": "Mentor sessions are typically held weekly.",
    "are mentors experienced": "Yes, mentors are experienced industry professionals.",
    "is mentorship free": "Yes, mentorship is included in the internship program at no extra cost.",
    "mentor support": "Athenura provides complete clarity and support regarding mentor guidance.",

    # ── Networking ────────────────────────────────────────────
    "will i get networking opportunities": "Yes, you can connect with professionals and fellow interns.",
    "are webinars conducted": "Yes, webinars are organized as part of the program.",
    "can i attend workshops": "Yes, workshops are included in the internship activities.",
    "can i build a portfolio": "Yes, the project work you do helps build a strong portfolio.",
    "is linkedin guidance provided": "Yes, guidance on building your LinkedIn profile is provided.",
    "can i join alumni network": "Yes, you can join the Athenura alumni network after completion.",
    "can i stay in touch after internship": "Yes — Athenura encourages networking; outstanding interns may be kept in touch for future openings or offered PPOs.",

    # ── Performance & Evaluation ──────────────────────────────
    "is performance feedback given": "Yes, regular feedback is given to help you improve.",
    "are rewards given": "Yes, top performers are recognized and rewarded.",
    "can i track progress": "Yes, you can track your progress through dashboards and reports.",
    "who evaluates performance": "Your performance is evaluated by Team Leaders, Department Heads, and the HR department.",
    "will i get performance report": "Yes, a performance report is shared with you.",
    "what criteria is used for evaluation": "Evaluation is based on task quality, deadline adherence, communication, teamwork, attendance, and contributions to projects.",
    "are tests included in evaluation": "Yes, tests and assignments are part of the evaluation.",
    "is attendance counted in evaluation": "Yes, attendance is factored into your overall evaluation.",
    "are top performers recognized": "Yes, top performers receive recognition and awards, and may receive LORs or placement offers.",
    "performance evaluation": "Performance evaluation is a key component of the internship framework.",
    "how can interns verify internship or check performance": "Interns can verify details and track performance at: www.athenura.in/internship/performance",
    "how do i get feedback": "Feedback is provided by your Team Head during reviews and project wrap-ups. Ask for periodic feedback from your reporting head if you want more guidance.",
    "how can i check my performance": "Interns can check their performance through the intern dashboard available in the internship portal at www.athenura.in/internship/performance.",
    "will low performance affect certificates": "Yes. If minimum performance criteria are not met, certificate eligibility may be affected.",
    "does performance affect placement": "Yes. Placement decisions are heavily influenced by performance evaluation.",
    "what is a media task": "A media task is an activity where interns share internship activities or related content on social media platforms.",
    "is a media task compulsory": "In many internship programs media tasks are required as part of participation and engagement tracking.",
    "what happens if i miss a media task": "Missing media tasks may reduce your performance score and impact evaluation.",
    "where do i submit media proof": "Media proof must be uploaded on the internship portal as instructed.",
    "if i delete my media post will it affect certification": "Yes. If the post is removed before verification it may affect evaluation and certification.",
    "how is intern performance evaluated": "Evaluation is based on attendance, task completion, media tasks, participation, and team collaboration.",
    "who reviews intern performance": "Performance is reviewed by Team Leaders, Department Heads, and the HR department.",

    # ── Verification Portal ───────────────────────────────────
    "what is the intern verification portal": "The Intern Verification Portal is an official tool on the Athenura website where interns can securely check their internship status, performance records, and completion validity using their Unique Verification ID.",
    "where can i access the verification portal": "You can access it at https://www.athenura.in/internship/performance under the 'Intern Verification' section.",
    "how is verification done": "Verification is done only through your Unique Verification ID assigned at the time of certificate generation or post-verification by HR.",
    "should i share my unique verification id with others": "Sharing your Unique Verification ID is not encouraged. Only share it when an institution, employer, or authorized person specifically asks for verification.",
    "what information can i see on the verification portal": "You can view your name, internship domain, internship duration, completion status, verification status, certificate ID (if issued), and performance summary.",
    "who can verify my internship through the portal": "Only individuals who have your Unique Verification ID can check your verification status. Without the ID, no one can access your information.",
    "i completed my internship but my verification is not showing": "If your status is not updated, contact the HR team at hr.athenura@gmail.com with your registered email, name, and internship details for manual updating.",
    "can i check whether my internship certificate is genuine using the portal": "Yes — enter your Unique Verification ID on the portal to confirm the authenticity of your certificate.",
    "will my college or employer need my unique verification id": "Yes, if your college or employer wants to verify your internship, you must provide your Unique Verification ID.",
    "can i update my details on the verification portal": "No, interns cannot update their own verification information. For changes or corrections, contact HR.",
    "i lost my unique verification id what do i do": "Contact HR at hr.athenura@gmail.com with your registered email to retrieve your ID.",
    "can someone misuse my unique verification id": "No personal information is exposed. The ID only shows internship verification details. Still, sharing your ID only when required is recommended.",
    "does the verification portal show my performance rating": "Yes — performance indicators or remarks may be shown depending on your department's evaluation system.",
    "will the portal show if i was terminated": "The portal only shows verified details. If you are terminated, the status will be shown as terminated.",
    "can i check my attendance through the verification portal": "No — attendance logs are handled internally. The portal only verifies internship authenticity and performance summaries.",
    "when will my unique verification id be generated": "IDs are generated after successful completion, verification, and internal approval.",

    # ── Planner & Schedule ────────────────────────────────────
    "what is the planner page on the website": "The Planner is a centralized schedule dashboard where interns can check all upcoming meetings, domain sessions, workshops, deadlines, and weekly activities.",
    "where can i access the planner": "The Planner is available at https://www.athenura.in/planner on the official Athenura website.",
    "what events are shown in the planner": "Daily meetings, domain-specific sessions, weekly club meetings, guest lectures, project deadlines, and special events or announcements.",
    "how often is the planner updated": "The Planner is updated regularly by the management/HR team whenever new meetings, events, or deadlines are added or modified.",
    "can i access the planner without logging in": "The Planner is publicly visible, but full meeting details or internal notes may be accessible only to active interns.",
    "what if i miss a meeting shown on the planner": "Missing scheduled meetings without approved leave may be counted as absence or lead to disciplinary action depending on your department policy.",
    "can i download or export the schedule from the planner": "You can sync your phone calendar with it or use the link available in your group description.",
    "if i change domains will my meeting schedule change in the planner": "The Planner is centralized so you can check every meeting at one place, so there is no need to update that separately.",
    "who maintains the planner": "The Planner is maintained by the Athenura admin/HR team, with inputs from Team Heads and Managers.",
    "is the planner available after i complete my internship": "The Planner is active for current interns. Once your internship ends, you may not see future schedules but can still view public information.",
    "can non-intern users access the planner": "Yes, but they will only see the general schedule, not department-specific internal details.",
    "why is my planner not showing updated meetings": "Try refreshing the page. If still outdated, your Team Head may not have updated the details yet.",
    "will the planner show national holidays or company holidays": "Yes — major holidays or days with no meetings will usually be reflected.",
    "can i rely fully on the planner for attendance and tasks": "The Planner helps with scheduling, but you must still follow instructions from your Team Head and official HR communications.",

    # ── Privacy & Security ────────────────────────────────────
    "is my data safe": "Yes, Athenura ensures complete data privacy and security.",
    "will my information be shared": "No, your information will not be shared without your consent.",
    "is platform secure": "Yes, all systems and platforms used are secure.",
    "do i need login credentials": "Yes, secure login credentials are provided upon enrollment.",
    "can i reset password": "Yes, a password reset option is available on the login page.",
    "privacy protection": "All interns must comply with Athenura's privacy protection policies.",
    "data protection": "Athenura ensures professional standards in all matters related to data protection.",

    # ── Confidentiality ───────────────────────────────────────
    "confidentiality": "All interns must comply with Athenura's confidentiality policies as shared during induction.",
    "is confidentiality mandatory": "Yes, maintaining confidentiality is mandatory for all interns.",
    "how important is confidentiality at athenura": "Confidentiality plays a key role in intern evaluation and development. Support and mentorship are provided concerning confidentiality.",
    "what are the rules for confidentiality": "Interns are expected to follow all rules related to confidentiality.",
    "can i share client data or company files outside athenura": "No — client or company data must not be shared externally. Breach of confidentiality may lead to legal action.",
    "who owns the work i create during the internship": "Work produced for Athenura during the internship is company property. Interns should assume IP/rights belong to the company.",
    "can i work on personal projects during the internship": "Personal projects must not use company resources or conflict with your Athenura duties. Disclose any potential conflicts to your Team Head.",

    # ── Code of Conduct ───────────────────────────────────────
    "code of conduct": "Interns are expected to follow all rules related to professional code of conduct.",
    "is there any policy regarding code of conduct": "Yes, the code of conduct is shared during induction and must be strictly followed.",
    "what are the professional conduct expectations": "Maintain discipline, respect all team members, meet deadlines, and communicate professionally. Disrespect or unprofessional behaviour may lead to disciplinary measures.",
    "is there a dress code": "Professional dress is required for official meetings. Follow the team's instructions for virtual or in-person sessions.",
    "can i post about athenura on social media": "Do not post company-related content publicly without prior approval. Negative or unapproved posts are subject to disciplinary action.",
    "can i create unofficial groups using the company name": "No. Unapproved groups using the company name or logo are not permitted.",

    # ── Task Submission ───────────────────────────────────────
    "task submission": "Task submission is a mandatory component of the internship framework.",
    "are assignments compulsory": "Yes, all assignments and tasks must be completed.",
    "are assignments practical": "Yes, assignments are hands-on and project-based.",
    "will i get deadlines for assignments": "Yes, clear deadlines are provided for all assignments.",
    "what are the expectations around deadlines": "Deadline commitment is non-negotiable — complete tasks on or before deadlines. Missing multiple deadlines may result in leave deductions or disciplinary action.",
    "what happens if i miss deadlines": "Missing 2 consecutive deadlines may lead to a 1-day leave deduction; missing 3 or more deadlines can result in termination as per policy.",
    "what is the file or assignment submission process": "Submit assignments via the official channel/platform specified by your Team Head (e.g., company Drive, project management tool, or email). Follow naming and versioning guidelines.",
    "can i reuse content from the web": "All work must be original and plagiarism-free. Copying others' work is prohibited and may lead to termination.",
    "how are creative assignments judged": "On originality, alignment with the brief, quality, and meeting deadlines. Seek clarifications before starting if the brief is unclear.",

    # ── Termination Policy ────────────────────────────────────
    "termination policy": "The termination policy is structured according to Athenura's official internship guidelines.",
    "how important is termination policy": "Termination policy is an important component of the internship framework.",
    "what actions can lead to termination": "Behavioural issues, repeated missed deadlines, unauthorized leave, serious misconduct, or breach of confidentiality can lead to immediate termination. Terminated candidates cannot reapply for 6 months.",
    "is there a warning system": "Yes — usually warnings are issued before major action, though serious violations may lead to immediate termination.",
    "will my college be informed if i am terminated": "In cases of unauthorized prolonged absence or serious misconduct, Athenura may notify your college.",
    "what legal steps does athenura take for serious violations": "For breaches such as data leakage or serious misconduct, Athenura may initiate legal action as described in the disciplinary policy.",

    # ── Exit & Resignation ────────────────────────────────────
    "is there a notice period during internship": "A 1-month notice period is required for proper exit and handover.",
    "what is the resignation notice period": "The notice period is usually 30 days depending on the internship policy.",
    "is the notice period compulsory": "Yes, unless HR grants special permission.",
    "can i resign immediately": "Immediate resignation may require HR approval.",
    "will i receive a certificate after resignation": "Yes, if internship requirements and performance criteria are fulfilled.",
    "what is the exit process for resignation": "Submit resignation, serve the notice period, complete tasks, HR verification, and then receive exit documents.",
    "project allocation": "Project allocation is managed under company supervision with structured processes.",
    "how does athenura handle project allocation": "Projects are allocated based on your domain and skill set.",

    # ── Working Hours ─────────────────────────────────────────
    "working hours": "Working hours are structured according to Athenura's official internship policies.",
    "what are the working hours": "Athenura follows a 5-day week (Monday–Friday). Typical virtual timing is flexible based on domain. Weekends are generally off.",
    "can i work remotely": "The program is virtual-first. Follow the set timings, attend meetings, and maintain availability as required by your domain/team.",

    # ── Completion ────────────────────────────────────────────
    "what are completion criteria": "You must complete all tasks, projects, and assessments to receive the certificate.",
    "do i need to submit projects": "Yes, project submission is mandatory for completion.",
    "will incomplete work affect certificate": "Yes, incomplete submissions may affect your certificate eligibility.",
    "is extension allowed": "Yes, extension is allowed in special cases with prior approval.",
    "what happens after completion": "After completion, you receive a certificate and can apply for job opportunities.",
    "will i get completion email": "Yes, a confirmation email is sent upon successful completion.",

    # ── Feedback ──────────────────────────────────────────────
    "can i give feedback": "Yes, feedback is actively encouraged from all interns.",
    "is feedback reviewed": "Yes, all feedback is reviewed for program improvement.",
    "is feedback anonymous": "Anonymous feedback options are sometimes available.",
    "can i give suggestions": "Yes, suggestions are always welcome.",

    # ── Platform & Access ─────────────────────────────────────
    "which platform is used": "Online learning and collaboration platforms are used during the internship.",
    "is platform user friendly": "Yes, the platforms are easy to use.",
    "can i access on mobile": "Yes, the platforms are mobile-friendly.",
    "is login required": "Yes, you need to log in with provided credentials.",

    # ── Communication & Notifications ─────────────────────────
    "will i get notifications": "Yes, updates and announcements are sent via email or the platform.",
    "can i contact support": "Yes, the support team is available to help you.",
    "is there a help desk": "Yes, a help desk is available for intern support.",
    "is support responsive": "Yes, the support team responds promptly.",
    "are faqs available": "Yes, FAQs are available on the Athenura website for guidance.",
    "how do i get notified about daily meetings": "Notifications are usually shared through official communication groups, portal notifications, HR announcements, or Team Leader messages.",
    "where is the meeting link shared": "Meeting links are generally shared through email, official groups, or intern dashboard announcements.",
    "why didnt i receive a meeting notification": "Possible reasons include incorrect contact details, technical email issues, not being added to the communication group, or spam filtering.",
    "can i get a meeting recording": "If the meeting was recorded, the recording may be shared depending on company policy and meeting type.",
    "how can i add a meeting to my calendar": "Open the meeting invitation and select the Add to Calendar option to save the reminder.",
    "what happens if i join a meeting late": "Inform your Team Leader and continue participating. Frequent late attendance may affect performance evaluation.",
    "will i be marked absent if i miss the meeting": "Yes. Missing a meeting without a valid reason may result in absent attendance.",
    "can i request a meeting with athenura": "Yes, meetings can be scheduled by contacting the team at +91 98350 51934 or via email.",

    # ── Roles & Domains ───────────────────────────────────────
    "what roles are available": "Various technical and non-technical roles are available.",
    "can i choose domain": "Yes, you can choose your domain based on your interest and skills.",
    "will i get role description": "Yes, a clear role description is provided before the internship starts.",

    # ── Career Growth ─────────────────────────────────────────
    "will i grow professionally": "Yes, you will experience significant professional growth through real-world exposure.",
    "will i gain confidence": "Yes, hands-on work and mentorship help build confidence.",
    "is it useful for interviews": "Yes, internship experience is very useful for job interviews.",
    "can i build contacts": "Yes, networking with professionals and peers is encouraged.",
    "can i recommend to others": "Yes, you can refer friends and colleagues to the program.",
    "will i enjoy learning": "Yes, the program is designed to be engaging and interactive.",
    "is workload manageable": "Yes, the workload is designed to be manageable for interns.",
    "can i learn at my own pace": "Partially yes — some flexibility in pace is allowed within deadlines.",
    "will i build a portfolio": "Yes, project-based work helps you build a strong portfolio.",
    "is this useful for future": "Yes, the internship is highly useful for your future career.",
    "will i face challenges": "Yes, challenges are part of the learning experience and help you grow.",
    "can i overcome difficulties": "Yes, mentors and the support team are there to help you through difficulties.",

    # ── Post Internship ───────────────────────────────────────
    "will i get support after internship": "Yes, career support and opportunities are shared even after the internship.",
    "can i stay connected after internship": "Yes, you can stay connected through the alumni community.",
    "are alumni groups available": "Yes, an alumni network exists for past interns.",
    "will i get updates after internship": "Yes, future opportunities and updates are shared with alumni.",

    # ── Soft Skills ───────────────────────────────────────────
    "will leadership skills improve": "Yes, leadership opportunities are available for high-performing interns.",
    "will i improve communication skills": "Yes, regular team interactions and presentations improve communication skills.",
    "are soft skills evaluated": "Yes, soft skills are part of the overall evaluation.",
    "are presentations required": "Yes, presentations may be required depending on the project.",

    # ── Leaves & Holidays ─────────────────────────────────────
    "how many official holidays": "Interns receive 4 official holidays during the internship period.",
    "are emergency leaves allowed": "Yes, 2 emergency leaves are allowed during the internship.",
    "how to apply for leave": "Apply for leave through the official portal: www.athenura.in/internship/leave",

    # ── Verification ──────────────────────────────────────────
    "how to verify internship": "Internship details can be verified at: www.athenura.in/internship/performance",
    "how to check performance": "Performance can be tracked at: www.athenura.in/internship/performance",

    # ── Troubleshooting ───────────────────────────────────────
    "i forgot my portal password what should i do": "Use the password recovery option on the login page. A reset link will be sent to your registered email address.",
    "what should i do if the portal is not working": "Check your internet connection, refresh the page, clear browser cache, or try another browser.",
    "why is my attendance not updating": "Attendance issues may occur due to network interruptions, system delays, or incorrect login details.",
    "why is my dashboard not loading": "This may occur due to browser compatibility issues or server delays. Try refreshing or switching browsers.",
    "why is the certificate portal not opening": "It may be due to server load, maintenance activity, or browser compatibility issues.",
    "i cannot upload my task what should i do": "Check file size limits, supported file format, and internet connection before uploading.",
    "why did i not receive the verification email": "Check your spam folder, verify your email address, and check email storage limits.",

    # ── Services for Clients ──────────────────────────────────
    "what services does athenura offer to clients": "Athenura offers training, internships, project development, and digital services including web development, digital marketing, branding, graphic design, video editing, UI/UX, and AI solutions.",
    "do you work with startups or only established brands": "We work with startups, SMEs, and enterprise brands. Our services are flexible and customizable for all stages of business growth.",
    "do you provide customized branding packages": "Yes — every branding package is customized based on your industry, target audience, creative needs, and brand personality.",
    "do you build websites too": "Yes, we build modern, responsive, and SEO-friendly websites including portfolio sites, e-commerce solutions, landing pages, and custom web platforms.",
    "do you provide content writing as well": "Yes — Athenura provides website content, ad copies, captions, blogs, scripts, and marketing content depending on the project.",
    "can athenura handle my company social media completely": "Yes — we offer complete social media management, including strategy, content creation, posting, ad campaigns, and analytics.",
    "can i get a complete branding website and marketing solution from one place": "Absolutely — we offer all-in-one brand transformation packages designed to build your brand from scratch.",
    "do you provide video ads or animated videos": "Yes — we create reels, product videos, promotional ads, explainer videos, motion graphics, and animated clips.",
    "do you create logos brand identity and brand guidelines": "Yes — we provide professional brand identity packages including logo, color schemes, typography, visual strategy, and brand guidelines.",
    "do you help in running advertisements": "Yes — we run and manage Meta (Facebook/Instagram), YouTube, and Google Ads along with analytical reporting.",
    "can you redesign my existing brand visuals": "Yes — we offer brand revamp and modernization services while maintaining your core identity.",
    "do you work internationally": "Yes — Athenura works with clients globally, especially in the international market.",
    "can i get a consultation call": "Yes — you can book a consultation call with our branding strategy team through our website or by contacting us directly.",
    "do you provide campaign strategy or only designs": "We provide full campaign strategy, design, and execution based on your goals.",
    "can you work with my internal marketing team": "Yes — we collaborate seamlessly with in-house teams to improve workflow and brand output.",
    "what are your pricing packages": "Pricing depends on project scope. After understanding your requirements, we provide a transparent and customized quote.",
    "do you offer monthly retainers": "Yes — we offer monthly retainer packages for brands needing continuous content or marketing support.",
    "can i get a detailed proposal for my project": "Absolutely — once we understand your brand needs, we prepare a detailed proposal including deliverables, timelines, and pricing.",
    "do you require advance payment": "Yes — most projects require a partial upfront payment to initiate work.",
    "do you sign contracts or mous": "Yes — for formal engagements, we provide a service agreement outlining scope, timelines, and responsibilities.",
    "can i negotiate pricing": "Pricing is structured based on workload and resources, but we can discuss flexible packages depending on requirements.",
    "do you offer refunds": "Refunds depend on the project stage and agreement terms. Details are mentioned in the service contract.",
    "how do i pay": "We accept payments through bank transfer, UPI, and other digital modes depending on region.",
    "do you provide emi or installment options": "For large projects, installment-based payment structures are available during milestone delivery.",
    "are taxes included in the package price": "Taxes are applied as applicable and will be mentioned clearly in the invoice.",
    "is there any hidden cost": "No — all costs are communicated upfront in the proposal.",
    "can i get a sample or trial before placing an order": "We don't provide trials, but you can check our portfolio and previous client work.",
    "can you work under an nda": "Yes — we respect privacy and can sign NDAs when required.",
    "will you give me the source files": "Yes — source files can be included depending on the package selected.",
    "how soon can you start on my project": "Once the proposal is approved and advance payment is completed, the project starts immediately.",
    "how long does it take to complete a branding project": "Depending on complexity, branding projects take 7–21 days typically.",
    "how long does website development take": "A standard website takes 15–30 days; advanced systems require more time.",
    "how many revisions will i get": "Revision limits depend on the selected package. We ensure you get exactly what your brand needs.",
    "what if i dont like the first draft": "We refine it based on your feedback until it aligns with your brand guidelines within the revision scope.",
    "do you follow a project timeline": "Yes — each project follows a structured timeline with milestones.",
    "how will i receive my files": "Files are shared via Google Drive or secure cloud links.",
    "will i get all versions of the logo": "Yes — you receive multiple formats (PNG, JPG, PDF, SVG) and color variations.",
    "do you support multilingual designs": "Yes — we can create designs in multiple languages based on your needs.",
    "can you manage my ads after delivering the creatives": "Yes — we offer ad management as an add-on service.",
    "will you maintain my website after delivery": "Yes — maintenance packages are available monthly or quarterly.",
    "can you design packaging as well": "Yes — packaging design is part of our brand identity services.",
    "do you deliver editable design files": "Editable files can be included in higher-tier packages.",
    "can i track project progress": "Yes — we share progress through regular updates, milestones, and meetings.",
    "what if my project needs urgent delivery": "We offer fast-track delivery at an additional charge.",
    "can i request old project files later": "Yes — we maintain backup for a specific duration; you can request past files.",
    "do you work on festive or campaign-based designs": "Yes — we provide seasonal, festive, and promotional content.",
    "can i request multiple services together": "Yes — bundled packages are available for branding + website + marketing.",
    "do you provide monthly content buckets": "Yes — monthly content and marketing calendars are provided under SMM packages.",
    "can i schedule meetings with your team during the project": "Yes — you can request meetings anytime based on your project requirements.",
    "how do you handle feedback and approvals": "Feedback is taken through scheduled calls, WhatsApp groups, or email, and approvals are documented before final delivery.",
    "how do i get support after project completion": "You can reach us through support or account managers assigned to your project.",
    "what is your response time": "Typical response time is 1–12 hours, depending on workload.",
    "do you provide 24/7 support": "Support hours depend on your package; premium support includes extended hours.",
    "will my project data remain confidential": "Yes — all files and business information remain strictly confidential.",
    "can i hire athenura for long-term brand management": "Yes — long-term partnerships and retainers are available.",
    "do you use ai tools in your work": "We combine human creativity with advanced tools for the best results — always ensuring originality.",
    "do you outsource your work": "No — our in-house experts handle all projects to maintain quality.",
    "will my brand have a dedicated manager": "Yes — for medium and large projects, a dedicated project manager is assigned.",
    "do you provide after-sales service": "Yes — depending on the agreement, you receive support for updates, edits, or maintenance.",
    "how do i report any issues": "You can report issues to your assigned manager or through official support channels.",
    "what industries have you worked with": "We've worked with food, tech, fashion, fitness, real estate, wellness, service-based, and product-based businesses.",
    "do you provide analytics for marketing campaigns": "Yes — monthly performance reports are included in SMM and ad management packages.",
    "how will i receive final files": "Through cloud delivery in high-quality export formats.",
    "do you offer seo services": "Yes — we provide on-page optimization, SEO-ready content, and website SEO setup.",
    "can athenura improve my existing website": "Yes — we can redesign, upgrade, optimize, or restructure your existing site.",
    "how do you ensure design originality": "All work is created from scratch following brand guidelines — we do not use plagiarized content.",
    "how do i start working with athenura": "Simply contact us through the website, share your requirements, and our team will schedule a call and prepare a proposal.",
    "how do you measure campaign success": "Through engagement, reach, keywords, CTR, conversions, and ROI metrics depending on your objectives.",
    "what makes athenura different from other branding agencies": "Our focus on personalized strategies, international-level quality, fast delivery, and complete brand ecosystem solutions sets us apart.",
    "does athenura provide digital marketing services": "Yes, Athenura supports digital marketing solutions including SEO, social media management, and ad campaigns.",
    "does athenura provide web development services": "Yes, Athenura provides web development support including frontend, backend, and full-stack solutions.",
    "can athenura help companies build software projects": "Yes, Athenura assists with software and technology projects through trained interns and development teams.",
    "does athenura offer ai-related services": "Yes, Athenura supports AI and data-related projects including machine learning and analytics.",
    "can athenura help startups with technical development": "Yes, Athenura helps startups build technical solutions through skilled interns and project teams.",
    "does athenura support research projects": "Yes, Athenura supports academic and industry research collaborations.",
    "can athenura provide skilled interns for projects": "Yes, trained interns can assist companies in projects across various domains.",
    "does athenura provide remote project teams": "Yes, remote collaboration is supported with distributed project teams.",
    "can companies hire interns through athenura": "Yes, companies can hire interns trained by Athenura through formal partnership agreements.",

    # ── Collaboration & Partnerships ──────────────────────────
    "how can companies collaborate with athenura": "Companies can collaborate with Athenura through partnerships, internships, and project development by contacting +91 98350 51934.",
    "does athenura work with industry partners": "Yes, Athenura collaborates with industry partners for training and projects.",
    "can startups collaborate with athenura": "Yes, startups can partner with Athenura for talent and development support.",
    "how can businesses partner with athenura": "Businesses can contact Athenura at +91 98350 51934 for collaboration opportunities.",
    "does athenura provide internship collaborations": "Yes, Athenura partners with companies for internship programs.",
    "can companies provide projects to athenura interns": "Yes, companies can provide real-world projects to Athenura interns as part of collaboration programs.",
    "does athenura collaborate with tech companies": "Yes, Athenura collaborates with various technology companies.",
    "can companies outsource projects to athenura": "Yes, Athenura supports project collaboration and development for external companies.",
    "does athenura provide workforce training for companies": "Yes, Athenura provides skill development programs for companies.",
    "how can organizations become partners with athenura": "Organizations can contact Athenura at +91 98350 51934 for partnership discussions.",
    "does athenura work with education institutions": "Yes, Athenura collaborates with educational institutions for internships and training programs.",
    "can colleges partner with athenura": "Yes, colleges can collaborate for internships and training programs.",
    "does athenura provide industry projects to students": "Yes, industry projects are part of Athenura programs for student interns.",
    "can training institutes collaborate with athenura": "Yes, training institutes can partner with Athenura for skill development programs.",
    "what makes athenura a good partner for companies": "Athenura provides trained talent, innovative solutions, and cost-effective collaboration options.",
    "does athenura support remote collaborations": "Yes, remote collaborations are supported with flexible communication and project management tools.",
    "is athenura open to international clients": "Yes, Athenura supports global collaborations with clients from various countries.",
    "can athenura help companies with innovation": "Yes, Athenura encourages innovative project development through skilled interns and expert guidance.",
    "does athenura support startups": "Yes, startups can collaborate with Athenura for talent acquisition and technical development support.",
    "what benefits do companies get by partnering with athenura": "Companies gain access to skilled interns, project support, cost-effective solutions, and innovative talent.",
    "does athenura provide cost-effective solutions": "Yes, Athenura offers affordable collaboration options without compromising on quality.",
    "can athenura help companies scale their projects": "Yes, Athenura supports scalable development with flexible team sizes and resource allocation.",
    "does athenura support digital transformation for companies": "Yes, Athenura assists companies with digital solutions including web development, automation, and AI integration.",
    "can companies build long-term partnerships with athenura": "Yes, long-term collaborations are encouraged with flexible engagement models and retainer options.",
    "which industries work with athenura": "Industries such as technology, marketing, education, healthcare, and e-commerce collaborate with Athenura.",
    "does athenura provide consultation for companies": "Yes, consultation is available for project and training collaborations.",
    "can companies discuss project ideas with athenura": "Yes, Athenura welcomes discussions about project ideas via phone or email.",
    "does athenura provide client support services": "Yes, Athenura provides support for clients and partners through dedicated channels.",
    "how quickly does athenura respond to client inquiries": "Athenura usually responds quickly through phone or official channels, typically within 1–12 hours.",
}


# ══════════════════════════════════════════════════════════════
#  SECTION 2 — KEYWORD RULE MAP
# ══════════════════════════════════════════════════════════════

KEYWORD_RULES: dict[str, list[str]] = {
    "what is athenura": ["what is athenura", "tell me about athenura", "about athenura", "explain athenura"],
    "is the internship paid": ["stipend", "paid", "salary", "earn money", "unpaid"],
    "will i receive a certificate": ["certificate", "cert", "completion certificate"],
    "is placement guaranteed": ["placement", "job after", "ppo", "pre placement"],
    "is attendance compulsory": ["attendance", "absent"],
    "will mentors guide projects": ["mentor", "guidance", "guide", "coach"],
    "is prior experience required": ["experience", "prior", "beginner", "fresher", "no experience"],
    "is my data safe": ["data safe", "privacy", "personal info", "data protection", "secure"],
    "what is the internship duration": ["internship duration", "how long is the internship", "duration of internship", "internship months", "internship weeks"],
    "what are completion criteria": ["completion", "complete", "finish", "end of internship"],
    "will i get networking opportunities": ["network", "connect", "professionals", "linkedin"],
    "how many holidays are provided": ["how many holidays", "official holidays", "days off", "holiday during internship", "number of holidays", "holidays provided"],
    "are emergency leaves available": ["are emergency", "how many leaves", "eligible for leave", "2 emergency"],
    "how can interns apply for emergency leave": ["emergency leave", "apply for leave", "leave portal", "urgent leave"],
    "what is the office address of athenura": ["office address", "address", "location", "where is athenura", "noida", "sector 62"],
    "what is the contact number of athenura": ["contact number", "phone number", "call athenura", "+91", "98350"],
    "what is the official hr contact email": ["hr email", "hr contact", "email hr", "contact hr"],
    "what is athenura official website": ["official website", "website", "athenura.in", "web address"],
    "what is the company hierarchy at athenura": ["hierarchy", "reporting structure", "company structure", "who is above"],
    "what internship domains are available at athenura": ["domains available", "which domain", "internship domain", "fields available"],
    "what is the planner page on the website": ["planner", "schedule dashboard", "meeting schedule"],
    "what is the intern verification portal": ["verification portal", "verify internship", "unique verification id", "intern portal"],
    "what is a media task": ["media task", "social media task", "post on social media"],
    "what actions can lead to termination": ["termination", "fired", "expelled", "removed from internship"],
    "is there a notice period during internship": ["notice period", "resignation", "how to resign", "quit internship"],
    "how many leaves are allowed during the internship": ["how many leaves", "leave quota", "total leaves allowed"],
    "what is a ppo": ["what is ppo", "pre placement offer", "ppo meaning"],
    "how can companies collaborate with athenura": ["company collaborate", "partner with athenura", "business partner"],
    "do you build websites too": ["build website", "web development", "website development", "website service"],
    "do you provide video ads or animated videos": ["video ads", "animated videos", "reels", "promotional video"],
    "what are your pricing packages": ["pricing", "price", "cost", "how much", "charges"],
    "can i get a consultation call": ["consultation", "consultation call", "discuss project", "book call"],
    "do you offer seo services": ["seo", "search engine optimization", "seo service"],
    "can athenura handle my company social media completely": ["social media management", "smm", "manage social media"],
    "what makes athenura different from other branding agencies": ["different from others", "unique about athenura", "why athenura", "athenura advantage"],
    "what is the saturday club meeting": ["saturday club", "club meeting", "weekend session", "guest speaker"],
    "what happens if i miss deadlines": ["miss deadline", "late submission", "deadline penalty", "deadline missed"],
    "can i share client data or company files outside athenura": ["share client data", "share company files", "data leak", "confidential data"],
    "who owns the work i create during the internship": ["work ownership", "ip rights", "who owns project", "intellectual property"],
    "i forgot my portal password what should i do": ["forgot password", "reset password", "password recovery", "login issue"],
    "can i request change of domain mid-internship": ["change domain", "switch domain", "change internship field"],
}


# ══════════════════════════════════════════════════════════════
#  SECTION 3 — TF-IDF ENGINE
# ══════════════════════════════════════════════════════════════

STOP_WORDS: set[str] = {
    "a", "an", "the", "is", "are", "do", "does", "can", "i", "you", "at",
    "in", "on", "for", "to", "of", "and", "or", "any", "there", "will",
    "how", "what", "when", "where", "why", "who", "which", "my", "me",
    "was", "were", "be", "been", "being", "have", "has", "had", "by",
    "its", "this", "that", "these", "those", "not", "no", "about", "with",
    "from", "as", "if", "it", "we", "they", "our", "your", "their",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into tokens, remove stop words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]


def build_tfidf_index(qa: dict[str, str]) -> tuple[dict[str, float], list[tuple[str, dict[str, float]]]]:
    """
    Build a TF-IDF index over all question keys.
    Returns (idf_scores dict, doc_vectors list of (key, vector)).
    """
    docs: list[str] = list(qa.keys())
    N: int = len(docs)

    df: defaultdict[str, int] = defaultdict(int)
    tokenized_docs: list[list[str]] = []
    for doc in docs:
        tokens = tokenize(doc)
        tokenized_docs.append(tokens)
        for term in set(tokens):
            df[term] += 1

    idf: dict[str, float] = {}
    for term, freq in df.items():
        idf[term] = math.log((N + 1) / (freq + 1)) + 1

    doc_vectors: list[tuple[str, dict[str, float]]] = []
    for i, tokens in enumerate(tokenized_docs):
        tf: defaultdict[str, float] = defaultdict(float)
        for t in tokens:
            tf[t] += 1
        total: int = len(tokens) if tokens else 1
        vec: dict[str, float] = {}
        for term, count in tf.items():
            vec[term] = (count / total) * idf.get(term, 1.0)
        norm: float = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        for k in vec:
            vec[k] /= norm
        doc_vectors.append((docs[i], vec))

    return idf, doc_vectors


def cosine_similarity(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    """Dot product of two normalized sparse vectors."""
    score: float = 0.0
    for term, val in vec1.items():
        if term in vec2:
            score += val * vec2[term]
    return score


def query_vector(text: str, idf: dict[str, float]) -> dict[str, float]:
    """Build a normalized TF-IDF vector for a query string."""
    tokens = tokenize(text)
    if not tokens:
        return {}
    tf: defaultdict[str, float] = defaultdict(float)
    for t in tokens:
        tf[t] += 1
    total: int = len(tokens)
    vec: dict[str, float] = {}
    for term, count in tf.items():
        vec[term] = (count / total) * idf.get(term, 0.5)
    norm: float = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    for k in vec:
        vec[k] /= norm
    return vec


# ══════════════════════════════════════════════════════════════
#  SECTION 4 — RESPONSE LOGIC
# ══════════════════════════════════════════════════════════════

def normalize(text: str) -> str:
    """Normalize user input for matching."""
    text = text.lower().strip()
    text = re.sub(r"[?!.,;:]+$", "", text).strip()
    return text


def handle_small_talk(text: str) -> Optional[str]:
    """Layer 1: Greetings, farewells, thanks."""
    t = normalize(text)
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon", "hii", "hlo"]
    thanks = ["thanks", "thank you", "thx", "thankyou", "ty"]
    byes = ["bye", "goodbye", "see you", "see ya", "exit", "quit", "cya"]

    for g in greetings:
        if t.startswith(g):
            return "Hello! 👋 Welcome to Athenura Chatbot. How can I help you today?"
    for b in byes:
        if b in t:
            return "QUIT"
    for thank in thanks:
        if thank in t:
            return "You're welcome! 😊 Feel free to ask anything else about Athenura."
    return None


def handle_datetime(text: str) -> Optional[str]:
    """Layer 2: Current time or date."""
    t = normalize(text)
    if "time" in t and ("current" in t or "what" in t or t == "time"):
        return f"🕐 Current time: {datetime.datetime.now().strftime('%I:%M:%S %p')}"
    if "date" in t and ("today" in t or "what" in t or "current" in t or t == "date"):
        return f"📅 Today's date: {datetime.datetime.now().strftime('%d %B %Y')}"
    return None


def handle_help(text: str) -> Optional[str]:
    """Layer 3: Help command."""
    t = normalize(text)
    if t in ("help", "?", "options", "menu", "what can you do"):
        return (
            "📋 You can ask me about:\n"
            "   • Athenura — company info, address, contact & services\n"
            "   • Internship — eligibility, domains, duration, mode\n"
            "   • Stipend, Certificate & Placement\n"
            "   • Attendance, Leaves & Holidays\n"
            "   • Mentorship & Career Guidance\n"
            "   • Projects, Tools & Technologies\n"
            "   • Privacy, Data Protection & Policies\n"
            "   • PPO, LOR & Performance Evaluation\n"
            "   • Planner, Verification Portal & Troubleshooting\n"
            "   • Client Services, Branding & Partnerships\n\n"
            "Type 'quit' to exit."
        )
    return None


def handle_keyword_rules(text: str) -> Optional[str]:
    """Layer 4: Fast keyword-based routing."""
    t = normalize(text)
    for qa_key, patterns in KEYWORD_RULES.items():
        for pattern in patterns:
            if pattern in t:
                return QA_DICT.get(qa_key)
    return None


def handle_exact_match(text: str) -> Optional[str]:
    """Layer 5: Direct dictionary key lookup."""
    key = normalize(text)
    return QA_DICT.get(key)


def handle_tfidf(
    text: str,
    idf: dict[str, float],
    doc_vectors: list[tuple[str, dict[str, float]]],
    threshold: float = 0.30
) -> Optional[str]:
    """Layer 6: TF-IDF cosine similarity — ML matching."""
    qvec = query_vector(text, idf)
    if not qvec:
        return None

    best_score: float = 0.0
    best_key: Optional[str] = None

    for doc_key, dvec in doc_vectors:
        score = cosine_similarity(qvec, dvec)
        if score > best_score:
            best_score = score
            best_key = doc_key

    if best_score >= threshold and best_key is not None:
        return QA_DICT.get(best_key)
    return None


def handle_partial_keyword(text: str) -> Optional[str]:
    """Layer 7: Fallback keyword overlap scoring."""
    tokens = set(tokenize(text))
    if not tokens:
        return None

    best_score: float = 0.0
    best_answer: Optional[str] = None

    for question, answer in QA_DICT.items():
        q_tokens = set(tokenize(question))
        if not q_tokens:
            continue
        overlap = tokens & q_tokens
        score = len(overlap) / len(tokens | q_tokens)
        if score > best_score:
            best_score = score
            best_answer = answer

    if best_score >= 0.25:
        return best_answer
    return None


def get_response(
    text: str,
    idf: dict[str, float],
    doc_vectors: list[tuple[str, dict[str, float]]]
) -> str:
    """
    Master response function — runs through all layers in order:
    1. Small talk  2. Date/Time  3. Help  4. Keywords
    5. Exact match  6. TF-IDF ML  7. Partial overlap  8. Fallback
    """
    reply: Optional[str]

    reply = handle_small_talk(text)
    if reply:
        return reply

    reply = handle_datetime(text)
    if reply:
        return reply

    reply = handle_help(text)
    if reply:
        return reply

    reply = handle_keyword_rules(text)
    if reply:
        return reply

    reply = handle_exact_match(text)
    if reply:
        return reply

    reply = handle_tfidf(text, idf, doc_vectors)
    if reply:
        return reply

    reply = handle_partial_keyword(text)
    if reply:
        return reply

    return (
        "🤔 I'm sorry, I couldn't find an answer to that.\n"
        "Please try rephrasing your question, or type 'help' to see what I can answer.\n"
        "You can also visit: www.athenura.in for more information."
    )


# ══════════════════════════════════════════════════════════════
#  SECTION 5 — MAIN CHAT LOOP
# ══════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 62)
    print("  🤖  ATHENURA CHATBOT  —  AI/ML Enhanced")
    print("  💡  Type 'help' to see topics  |  'quit' to exit")
    print("=" * 62)
    print()

    print("⚙️  Initializing AI engine...", end=" ", flush=True)
    idf, doc_vectors = build_tfidf_index(QA_DICT)
    print("Ready! ✅\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBot: Goodbye! 👋 Have a great day!")
            break

        if not user_input:
            continue

        response = get_response(user_input, idf, doc_vectors)

        if response == "QUIT":
            print("Bot: Goodbye! 👋 Thank you for using Athenura Chatbot. Have a great day!")
            break

        print(f"Bot: {response}\n")


# ══════════════════════════════════════════════════════════════
#  SECTION 6 — FLASK API WRAPPER
# ══════════════════════════════════════════════════════════════

# Build TF-IDF once globally (for Flask API use)
idf_global: dict[str, float]
doc_vectors_global: list[tuple[str, dict[str, float]]]
idf_global, doc_vectors_global = build_tfidf_index(QA_DICT)


def chatbot_response(user_input: str) -> str:
    """Wrapper function for Flask API."""
    if not user_input:
        return "Please enter a valid message."
    return get_response(user_input, idf_global, doc_vectors_global)


if __name__ == "__main__":
    main()