# parsers.py

import re
from datetime import datetime
from pathlib import Path

def parse_tasks_and_expenses(note_text):
    """
    Extract tasks and expenses from a single note.
    """
    tasks = re.findall(r"- \[([ xX])\] (.+)", note_text)
    expenses = re.findall(r"- ₹?(\d+(?:,\d{3})*\.?\d*) - (.+)", note_text)

    task_data = []
    for status, task in tasks:
        task_data.append({
            "task": task.strip(),
            "status": "done" if status.strip().lower() == "x" else "todo"
        })

    expense_data = []
    for amount, category in expenses:
        expense_data.append({
            "category": category.strip(),
            "amount": float(amount.replace(",", ""))
        })

    return task_data, expense_data

def aggregate_notes(folder_path="notes"):
    """
    Load all .md notes, parse and return structured tasks and expenses.
    """
    all_tasks, all_expenses = [], []
    for file in Path(folder_path).glob("**/*.md"):
        with open(file, "r") as f:
            content = f.read()
        tasks, expenses = parse_tasks_and_expenses(content)
        date_str = file.stem if file.stem[:4].isdigit() else None
        for task in tasks:
            task["date"] = date_str
        for expense in expenses:
            expense["date"] = date_str
        all_tasks.extend(tasks)
        all_expenses.extend(expenses)
    return all_tasks, all_expenses
