from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from modules.segmentation import DetectedCircle, detect_coins, draw_circles


CANVAS_WIDTH = 960
CANVAS_HEIGHT = 640


class EuroVisionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("CoinScope - Detection de pieces")
        self.root.geometry("1360x860")
        self.root.minsize(1180, 760)

        self.current_path: Path | None = None
        self.original_bgr = None
        self.annotated_bgr = None
        self.detected_circles: list[DetectedCircle] = []
        self._canvas_photo: ImageTk.PhotoImage | None = None

        self.status_var = tk.StringVar(value="Charge une image pour commencer.")
        self.image_info_var = tk.StringVar(value="Aucune image chargee")
        self.count_var = tk.StringVar(value="Pieces detectees : 0")

        self._build_layout()

    def _build_layout(self) -> None:
        self.root.configure(bg="#eef4f1")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Root.TFrame", background="#eef4f1")
        style.configure("Panel.TFrame", background="#fbfffd")
        style.configure("Header.TFrame", background="#0f3d3e")
        style.configure("Title.TLabel", background="#0f3d3e", foreground="#f4fbf8", font=("Avenir", 26, "bold"))
        style.configure("Muted.TLabel", background="#0f3d3e", foreground="#b7d4cf", font=("Avenir", 11))
        style.configure("PanelTitle.TLabel", background="#fbfffd", foreground="#0f3d3e", font=("Avenir", 13, "bold"))
        style.configure("Body.TLabel", background="#fbfffd", foreground="#284445", font=("Avenir", 11))
        style.configure("Primary.TButton", font=("Avenir", 11, "bold"))
        style.configure("Accent.TButton", font=("Avenir", 11, "bold"))

        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=18)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=5)
        root_frame.columnconfigure(1, weight=2)
        root_frame.rowconfigure(1, weight=1)

        header = ttk.Frame(root_frame, style="Header.TFrame", padding=18)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="CoinScope", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Interface locale pour tester la detection de pieces d'euros par transformee de Hough",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Label(
            header,
            text="Version experimentation",
            style="Muted.TLabel",
        ).grid(row=0, column=1, sticky="e")

        viewer_panel = ttk.Frame(root_frame, style="Panel.TFrame", padding=14)
        viewer_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 14))
        viewer_panel.columnconfigure(0, weight=1)
        viewer_panel.rowconfigure(1, weight=1)

        controls = ttk.Frame(viewer_panel, style="Panel.TFrame")
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        for column in range(4):
            controls.columnconfigure(column, weight=1)

        ttk.Button(controls, text="Choisir une image", command=self.open_image, style="Primary.TButton").grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )
        ttk.Button(controls, text="Analyser", command=self.run_detection, style="Accent.TButton").grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(controls, text="Voir l'original", command=self.show_original).grid(
            row=0, column=2, sticky="ew", padx=4
        )
        ttk.Button(controls, text="Exporter l'image", command=self.save_result).grid(
            row=0, column=3, sticky="ew", padx=(8, 0)
        )

        canvas_frame = tk.Frame(viewer_panel, bg="#d9ece7", highlightthickness=0)
        canvas_frame.grid(row=1, column=0, sticky="nsew")

        self.canvas = tk.Canvas(
            canvas_frame,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#cfe2db",
            bd=0,
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self._refresh_canvas)

        self.canvas.create_text(
            CANVAS_WIDTH // 2,
            CANVAS_HEIGHT // 2,
            text="Apercu en attente",
            fill="#356363",
            font=("Avenir", 20, "bold"),
            tags="placeholder",
        )

        side_panel = ttk.Frame(root_frame, style="Panel.TFrame", padding=16)
        side_panel.grid(row=1, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(3, weight=1)

        ttk.Label(side_panel, text="Session d'analyse", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(side_panel, textvariable=self.image_info_var, style="Body.TLabel", wraplength=300).grid(
            row=1, column=0, sticky="ew", pady=(10, 6)
        )
        ttk.Label(side_panel, textvariable=self.count_var, style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 12)
        )

        ttk.Label(side_panel, text="Cercles detectes", style="PanelTitle.TLabel").grid(
            row=3, column=0, sticky="nw"
        )

        self.details_text = tk.Text(
            side_panel,
            wrap="word",
            height=20,
            bg="#f3faf7",
            fg="#214243",
            relief="flat",
            font=("Menlo", 11),
            padx=10,
            pady=10,
        )
        self.details_text.grid(row=4, column=0, sticky="nsew", pady=(10, 12))
        self.details_text.insert("1.0", "Aucune detection pour l'instant.")
        self.details_text.configure(state="disabled")

        footer = ttk.Frame(root_frame, style="Root.TFrame")
        footer.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(14, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.status_var, style="Muted.TLabel").grid(row=0, column=0, sticky="w")

    def open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Choisir une image",
            initialdir=str(Path("dataset/images").resolve()),
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if not file_path:
            return

        image = cv2.imread(file_path)
        if image is None:
            messagebox.showerror("Image illisible", f"Impossible de lire :\n{file_path}")
            return

        self.current_path = Path(file_path)
        self.original_bgr = image
        self.annotated_bgr = None
        self.detected_circles = []

        height, width = image.shape[:2]
        self.image_info_var.set(f"{self.current_path.name}\nResolution : {width} x {height}")
        self.count_var.set("Pieces detectees : 0")
        self.status_var.set("Image chargee. Clique sur Analyser pour lancer la detection.")
        self._set_details("Aucune detection pour l'instant.")
        self._show_bgr_image(self.original_bgr)

    def run_detection(self) -> None:
        if self.original_bgr is None:
            messagebox.showinfo("Aucune image", "Charge d'abord une image.")
            return

        self.status_var.set("Analyse en cours...")
        self.root.update_idletasks()

        self.detected_circles = detect_coins(self.original_bgr)
        self.annotated_bgr = draw_circles(self.original_bgr, self.detected_circles)
        self.count_var.set(f"Pieces detectees : {len(self.detected_circles)}")
        self.status_var.set("Analyse terminee.")
        self._set_details(self._format_circles(self.detected_circles))
        self._show_bgr_image(self.annotated_bgr)

    def show_original(self) -> None:
        if self.original_bgr is None:
            return

        self.status_var.set("Affichage de l'image originale.")
        self._show_bgr_image(self.original_bgr)

    def save_result(self) -> None:
        if self.annotated_bgr is None:
            messagebox.showinfo("Rien a enregistrer", "Lance d'abord la detection.")
            return

        default_name = "coinscope_resultat.jpg"
        if self.current_path is not None:
            default_name = f"{self.current_path.stem}_detected.jpg"

        target = filedialog.asksaveasfilename(
            title="Exporter l'image annotee",
            defaultextension=".jpg",
            initialfile=default_name,
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("Tous les fichiers", "*.*"),
            ],
        )
        if not target:
            return

        success = cv2.imwrite(target, self.annotated_bgr)
        if not success:
            messagebox.showerror("Echec", f"Impossible d'enregistrer :\n{target}")
            return

        self.status_var.set(f"Resultat exporte dans {target}")

    def _refresh_canvas(self, _event: tk.Event) -> None:
        if self.annotated_bgr is not None:
            self._show_bgr_image(self.annotated_bgr)
        elif self.original_bgr is not None:
            self._show_bgr_image(self.original_bgr)

    def _show_bgr_image(self, image_bgr) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        canvas_width = max(self.canvas.winfo_width(), 200)
        canvas_height = max(self.canvas.winfo_height(), 200)
        image.thumbnail((canvas_width - 16, canvas_height - 16), Image.Resampling.LANCZOS)

        self._canvas_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self._canvas_photo, anchor="center")

    def _set_details(self, text: str) -> None:
        self.details_text.configure(state="normal")
        self.details_text.delete("1.0", "end")
        self.details_text.insert("1.0", text)
        self.details_text.configure(state="disabled")

    def _format_circles(self, circles: list[DetectedCircle]) -> str:
        if not circles:
            return "Aucune piece detectee sur cette image."

        lines = []
        for index, circle in enumerate(circles, start=1):
            lines.append(
                f"{index:02d} | centre=({circle.x:4d}, {circle.y:4d}) | rayon={circle.radius:3d}px"
            )
        return "\n".join(lines)


def main() -> None:
    root = tk.Tk()
    app = EuroVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
