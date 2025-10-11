import tkinter as tk
from tkinter import ttk
import subprocess
import threading
import webbrowser
import os

# Create the main window
window = tk.Tk()
window.title("Liquid Node 2")
window.geometry("500x400")

# Apply themed styling
brand_background = "#0f172a"
surface_background = "#1e293b"
accent_color = "#38bdf8"
success_color = "#34d399"
error_color = "#f87171"
text_primary = "#e2e8f0"
text_muted = "#94a3b8"

window.configure(bg=brand_background)

style = ttk.Style()
style.theme_use("clam")

style.configure("Brand.TFrame", background=brand_background)
style.configure("Surface.TFrame", background=surface_background)

style.configure(
    "StatusInfo.TLabel",
    background=brand_background,
    foreground=accent_color,
    font=("Segoe UI", 12, "bold"),
)
style.configure(
    "StatusProgress.TLabel",
    background=brand_background,
    foreground=text_muted,
    font=("Segoe UI", 12, "bold"),
)
style.configure(
    "StatusSuccess.TLabel",
    background=brand_background,
    foreground=success_color,
    font=("Segoe UI", 12, "bold"),
)
style.configure(
    "StatusError.TLabel",
    background=brand_background,
    foreground=error_color,
    font=("Segoe UI", 12, "bold"),
)

style.configure(
    "Primary.TButton",
    font=("Segoe UI", 11, "bold"),
    background=accent_color,
    foreground=brand_background,
    padding=(14, 8),
    borderwidth=0,
)
style.map(
    "Primary.TButton",
    background=[("active", "#0ea5e9"), ("disabled", "#334155")],
    foreground=[("disabled", text_muted)],
)

style.configure(
    "Secondary.TButton",
    font=("Segoe UI", 11),
    background=surface_background,
    foreground=text_primary,
    padding=(12, 6),
)
style.map(
    "Secondary.TButton",
    background=[("active", "#334155"), ("disabled", "#1f2937")],
    foreground=[("disabled", text_muted)],
)

style.configure(
    "Command.TEntry",
    foreground=text_primary,
    fieldbackground=surface_background,
    font=("Segoe UI", 11),
)
style.map(
    "Command.TEntry",
    fieldbackground=[("disabled", "#1f2937")],
    foreground=[("disabled", text_muted)],
)

style.configure(
    "OutputInfo.TLabel",
    background=surface_background,
    foreground=text_primary,
    font=("Segoe UI", 10),
    padding=(8, 6),
)
style.configure(
    "OutputError.TLabel",
    background=surface_background,
    foreground=error_color,
    font=("Segoe UI", 10),
    padding=(8, 6),
)

main_frame = ttk.Frame(window, style="Brand.TFrame", padding="24 24 24 24")
main_frame.pack(fill="both", expand=True)

status_label = ttk.Label(
    main_frame,
    text="Click to create a server.",
    style="StatusInfo.TLabel",
)
status_label.pack(anchor="w", pady=(0, 20))

# Function to create and start the server
def server_make():
    def run_docker():
        try:
            status_label.config(text="Creating server...", style="StatusProgress.TLabel")
            window.update_idletasks()

            # Ensure the Dockerfile is created with the correct name in the current working directory
            dockerfile_path = "Dockerfile"
            with open(dockerfile_path, "w") as file:
                file.write("""\
# This is a Dockerfile for Debian
FROM debian:bullseye
RUN apt-get update
RUN apt-get install -y nginx
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
""")

            # Verify the Dockerfile exists before proceeding
            if not os.path.exists(dockerfile_path):
                status_label.config(text="Dockerfile not found!", style="StatusError.TLabel")
                return

            # Build Docker image
            subprocess.run(["docker", "build", "-t", "new-server", "."], check=True)
            # Run Docker container
            subprocess.run(["docker", "run", "-d", "-p", "8080:80", "--name", "liquid_server", "new-server"], check=True)

            status_label.config(
                text="Server started successfully on port 8080!",
                style="StatusSuccess.TLabel",
            )
            open_browser_button.config(state="normal")  # Enable the Open Browser button
            command_entry.config(state="normal")  # Enable the command input
            run_command_button.config(state="normal")

        except subprocess.CalledProcessError as e:
            status_label.config(text=f"Error: {e}", style="StatusError.TLabel")
        except Exception as e:
            status_label.config(text=f"Unexpected error: {e}", style="StatusError.TLabel")

    threading.Thread(target=run_docker, daemon=True).start()

# Function to open localhost in the browser
def open_browser():
    webbrowser.open("http://localhost:8080")

# Function to run a command inside the Docker container
def run_command():
    command = command_entry.get()
    if command:
        try:
            result = subprocess.run(["docker", "exec", "liquid_server", "bash", "-c", command], capture_output=True, text=True, check=True)
            output_label.config(text=f"Output:\n{result.stdout}", style="OutputInfo.TLabel")
        except subprocess.CalledProcessError as e:
            output_label.config(text=f"Error: {e.stderr}", style="OutputError.TLabel")

# Container for controls
controls_frame = ttk.Frame(main_frame, style="Surface.TFrame", padding="16 16 16 16")
controls_frame.pack(fill="x", pady=(0, 16))

# Create the server button
create_server_button = ttk.Button(
    controls_frame,
    text="Create a Server",
    command=server_make,
    style="Primary.TButton",
)
create_server_button.pack(fill="x")

# Create a button to open the server in a browser
open_browser_button = ttk.Button(
    controls_frame,
    text="Open Server in Browser",
    command=open_browser,
    state="disabled",
    style="Secondary.TButton",
)
open_browser_button.pack(fill="x", pady=(12, 0))

# Input field for commands
command_section = ttk.Frame(main_frame, style="Brand.TFrame")
command_section.pack(fill="x", pady=(0, 16))

command_label = ttk.Label(
    command_section,
    text="Enter command for server:",
    style="StatusProgress.TLabel",
)
command_label.pack(anchor="w", pady=(0, 8))

command_entry = ttk.Entry(
    command_section,
    width=40,
    state="disabled",
    style="Command.TEntry",
)
command_entry.pack(fill="x")

# Button to run the entered command
run_command_button = ttk.Button(
    command_section,
    text="Run Command",
    command=run_command,
    state="disabled",
    style="Primary.TButton",
)
run_command_button.pack(fill="x", pady=(12, 0))

# Label to show command output
output_container = ttk.Frame(main_frame, style="Surface.TFrame", padding="16 16 16 16")
output_container.pack(fill="both", expand=True)

output_label = ttk.Label(
    output_container,
    text="",
    style="OutputInfo.TLabel",
    anchor="nw",
    wraplength=420,
    justify="left",
)
output_label.pack(fill="both", expand=True)

# Start the Tkinter event loop
window.mainloop()

