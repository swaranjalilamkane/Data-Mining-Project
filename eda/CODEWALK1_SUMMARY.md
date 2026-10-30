# Checkpoint 4: Dataset Exploration

**Course:** Data Mining Techniques
**Name:** Swaranjali Kulbhushan Lamkane
**Dataset to Analyze:** StudentLife Dataset - Dartmouth College
**Source code Link:** https://github.com/swaranjalilamkane/Data-Mining-Project

I explored the StudentLife dataset. It was collected at Dartmouth College in spring 2013. It has data
from 49 students over the 10-week term, spread across about 1,500 files, with separate files for each
student. The data includes:

- **Phone sensors:** these don't save real sound or movement. They only save the phone's guess. For
  example, the phone listened every second and wrote down "silence", "voice", or "noise", but it never
  recorded the actual sound. That gives about 99 million audio rows and 23 million activity rows
  (still, walking, or running).
- **App data:** which app was on the screen, saved every 20 minutes (about 2 million rows).
- **SMS and call logs.**
- **EMA:** short surveys that popped up on the phone during the day, like "How stressed are you right
  now?" There are 26 types.
- **8 questionnaires** taken at the start and end of the term (depression, stress, loneliness, and
  others).
- **GPA** for 30 of the students.

I combined all of this into three tables:

1. **Hourly table:** one row for each student for each hour (77,616 rows).
2. **Daily table:** one row for each student for each day, together with their sleep and stress
   answers (3,234 rows).
3. **Student table:** one row for each student, with their questionnaire scores and GPA (49 rows).

## Preprocessing decisions

- The times in the data were in UTC, which is 4 hours ahead of Dartmouth time. I changed them to
  Dartmouth local time. Otherwise, a student using their phone at 10 pm would look like they were
  using it at 2 am.
- Some files had small problems: hidden characters at the start, extra spaces in column names, an
  extra comma at the end of each line, and empty lines between rows. The app files were also not in
  time order, and some audio readings were repeated (about 5%). I cleaned all of this, sorted the app
  data by time, and removed the repeats.
- The app data doesn't tell us when a student opened or closed the phone. It only saves which app was
  on the screen every 20 minutes. So I count how many times the app on the screen changed, and I use
  that as a sign the student was using the phone.
- If the phone recorded nothing for an hour, I keep that hour as "missing" and not as zero, because we
  don't know what happened. Also, 26 students' SMS files and 29 students' call files had no messages
  or calls saved at all, so I marked them as missing, not "0 texts".
- The stress answers were numbered confusingly: 4 and 5 meant "feeling good", not "more stressed". I
  renumbered them so 1 is great, and 5 is very stressed.

## References

1. StudentLife dataset: Wang, R., et al. (2014). StudentLife: Assessing mental health, academic
   performance and behavioral trends of college students using smartphones.
   https://doi.org/10.1145/2632048.2632054
2. García, S., Ramírez-Gallego, S., Luengo, J., Benítez, J. M., & Herrera, F. (2016). Big data
   preprocessing: Methods and prospects. Big Data Analytics, 1, Article 9.
   https://doi.org/10.1186/s41044-016-0014-0
