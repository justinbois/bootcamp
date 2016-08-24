import shutil
import re

lesson = re.compile('lessons.*html')
exercise = re.compile('exercises.*html')

with open('schedule.html', 'r') as f:
    for line in f:
        lesson_search = lesson.search(line)
        exercise_search = exercise.search(line)

        if lesson_search is not None:
            new_file = lesson_search.group()
            if 'l00' not in new_file:
                shutil.copyfile('404.html', lesson_search.group())

        if exercise_search is not None:
            shutil.copyfile('404.html', exercise_search.group())
