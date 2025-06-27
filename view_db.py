import tkinter as tk
from tkinter import ttk
import sqlite3

DB_PATH = "attendance_log.db"

def fetch_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, datetime FROM attendance ORDER BY datetime DESC")
    rows = c.fetchall()
    conn.close()
    return rows

def refresh_table(tree):
    for row in tree.get_children():
        tree.delete(row)
    for row in fetch_data():
        tree.insert("", "end", values=row)

def main():
    root = tk.Tk()
    root.title("Attendance Database Viewer")
    root.geometry("500x400")

    columns = ("ID", "Name", "Date/Time")
    tree = ttk.Treeview(root, columns=columns, show="headings")
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, anchor=tk.CENTER)
    tree.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    refresh_btn = tk.Button(root, text="Refresh", command=lambda: refresh_table(tree))
    refresh_btn.pack(pady=5)

    refresh_table(tree)
    root.mainloop()

if __name__ == "__main__":
    main()
