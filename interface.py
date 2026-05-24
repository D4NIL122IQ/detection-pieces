from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from modules.classification import ValeurPiece, classify_by_color_and_size, valeur_totale
from modules.classification2 import ValeurPieceHLS, classify_hls, valeur_totale_hls
from modules.classification3 import ValeurPieceFiltre, classify_filtres, valeur_totale_filtres
from modules.classification4 import ValeurPieceCombinee, classify_combine, valeur_totale_combinee
from modules.labelme_parser import load_labelme_annotation
from modules.segmentation import DetectedCircle, detect_coins, draw_circles

BDD_DIR = Path("dataset/BDD")

LABEL_COLORS = {
    "1cent": (0, 0, 255), "2cent": (0, 80, 255), "5cent": (0, 165, 255),
    "10cent": (0, 255, 255), "20cent": (0, 255, 128), "50cent": (0, 255, 0),
    "1euro": (255, 200, 0), "2euro": (255, 0, 0),
}


CANVAS_WIDTH = 960
CANVAS_HEIGHT = 640


class EuroVisionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Detection de pieces")
        self.root.geometry("1360x860")
        self.root.minsize(1180, 760)

        self.current_path: Path | None = None
        self.original_bgr = None
        self.annotated_bgr_hsv = None
        self.annotated_bgr_hls = None
        self.annotated_bgr_filtres = None
        self.annotated_bgr_combine = None
        self.detected_circles: list[DetectedCircle] = []
        self.valuations_hsv: list[ValeurPiece] = []
        self.valuations_hls: list[ValeurPieceHLS] = []
        self.valuations_filtres: list[ValeurPieceFiltre] = []
        self.valuations_combine: list[ValeurPieceCombinee] = []
        self._canvas_photo: ImageTk.PhotoImage | None = None

        self.status_var = tk.StringVar(value="Charge une image pour commencer.")
        self.image_info_var = tk.StringVar(value="Aucune image chargee")
        self.count_var = tk.StringVar(value="Pieces detectees : 0")
        self.total_hsv_var = tk.StringVar(value="")
        self.total_hls_var = tk.StringVar(value="")
        self.total_filtres_var = tk.StringVar(value="")
        self.total_combine_var = tk.StringVar(value="")

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
        style.configure("Total.TLabel", background="#fbfffd", foreground="#0f6e47", font=("Avenir", 15, "bold"))

        root_frame = ttk.Frame(self.root, style="Root.TFrame", padding=18)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=5)
        root_frame.columnconfigure(1, weight=2)
        root_frame.rowconfigure(1, weight=1)

        header = ttk.Frame(root_frame, style="Header.TFrame", padding=18)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 14))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Détection de pièces", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Comparaison de la methode principale avec les variantes",
            style="Muted.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

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
        ttk.Button(controls, text="Voir le labeling", command=self.show_labeling).grid(
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

        # --- Panneau latéral avec onglets ---
        side_panel = ttk.Frame(root_frame, style="Panel.TFrame", padding=16)
        side_panel.grid(row=1, column=1, sticky="nsew")
        side_panel.columnconfigure(0, weight=1)
        side_panel.rowconfigure(3, weight=1)

        ttk.Label(side_panel, text="Session d'analyse", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(side_panel, textvariable=self.image_info_var, style="Body.TLabel", wraplength=300).grid(
            row=1, column=0, sticky="ew", pady=(10, 6)
        )
        ttk.Label(side_panel, textvariable=self.count_var, style="Body.TLabel").grid(
            row=2, column=0, sticky="w", pady=(0, 8)
        )

        # Notebook (onglets methode principale / variantes)
        self.notebook = ttk.Notebook(side_panel)
        self.notebook.grid(row=3, column=0, sticky="nsew", pady=(4, 0))
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_change)

        # Onglet principal
        tab_hsv = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_hsv, text="Analyse principale")
        tab_hsv.columnconfigure(0, weight=1)
        tab_hsv.rowconfigure(1, weight=1)

        ttk.Label(tab_hsv, textvariable=self.total_hsv_var, style="Total.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.details_hsv = tk.Text(
            tab_hsv, wrap="word", height=18, bg="#f3faf7", fg="#214243",
            relief="flat", font=("Menlo", 10), padx=10, pady=10,
        )
        self.details_hsv.grid(row=1, column=0, sticky="nsew")
        self.details_hsv.insert("1.0", "Aucune detection pour l'instant.")
        self.details_hsv.configure(state="disabled")

        # Onglet HLS
        tab_hls = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_hls, text="Analyse HLS")
        tab_hls.columnconfigure(0, weight=1)
        tab_hls.rowconfigure(1, weight=1)

        ttk.Label(tab_hls, textvariable=self.total_hls_var, style="Total.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.details_hls = tk.Text(
            tab_hls, wrap="word", height=18, bg="#f7f3fa", fg="#2d2143",
            relief="flat", font=("Menlo", 10), padx=10, pady=10,
        )
        self.details_hls.grid(row=1, column=0, sticky="nsew")
        self.details_hls.insert("1.0", "Aucune detection pour l'instant.")
        self.details_hls.configure(state="disabled")

        # Onglet Filtres
        tab_filtres = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_filtres, text="Analyse Filtres")
        tab_filtres.columnconfigure(0, weight=1)
        tab_filtres.rowconfigure(1, weight=1)

        ttk.Label(tab_filtres, textvariable=self.total_filtres_var, style="Total.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.details_filtres = tk.Text(
            tab_filtres, wrap="word", height=18, bg="#faf7f3", fg="#433421",
            relief="flat", font=("Menlo", 10), padx=10, pady=10,
        )
        self.details_filtres.grid(row=1, column=0, sticky="nsew")
        self.details_filtres.insert("1.0", "Aucune detection pour l'instant.")
        self.details_filtres.configure(state="disabled")

        # Onglet Combiné
        tab_combine = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab_combine, text="Combiné (HSV+Filtres+Bimétal)")
        tab_combine.columnconfigure(0, weight=1)
        tab_combine.rowconfigure(1, weight=1)

        ttk.Label(tab_combine, textvariable=self.total_combine_var, style="Total.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        self.details_combine = tk.Text(
            tab_combine, wrap="word", height=18, bg="#f3f7fa", fg="#1a2d43",
            relief="flat", font=("Menlo", 10), padx=10, pady=10,
        )
        self.details_combine.grid(row=1, column=0, sticky="nsew")
        self.details_combine.insert("1.0", "Aucune detection pour l'instant.")
        self.details_combine.configure(state="disabled")

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
        self.annotated_bgr_hsv = None
        self.annotated_bgr_hls = None
        self.annotated_bgr_filtres = None
        self.annotated_bgr_combine = None
        self.detected_circles = []
        self.valuations_hsv = []
        self.valuations_hls = []
        self.valuations_filtres = []
        self.valuations_combine = []

        height, width = image.shape[:2]
        self.image_info_var.set(f"{self.current_path.name}\nResolution : {width} x {height}")
        self.count_var.set("Pieces detectees : 0")
        self.total_hsv_var.set("")
        self.total_hls_var.set("")
        self.total_filtres_var.set("")
        self.total_combine_var.set("")
        self.status_var.set("Image chargee. Clique sur Analyser pour lancer la detection.")
        self._set_details(self.details_hsv, "Aucune detection pour l'instant.")
        self._set_details(self.details_hls, "Aucune detection pour l'instant.")
        self._set_details(self.details_filtres, "Aucune detection pour l'instant.")
        self._set_details(self.details_combine, "Aucune detection pour l'instant.")
        self._show_bgr_image(self.original_bgr)

    def run_detection(self) -> None:
        if self.original_bgr is None:
            messagebox.showinfo("Aucune image", "Charge d'abord une image.")
            return

        self.status_var.set("Analyse en cours...")
        self.root.update_idletasks()

        # Detection commune
        self.detected_circles = detect_coins(self.original_bgr)
        self.count_var.set(f"Pieces detectees : {len(self.detected_circles)}")

        # Classification principale
        self.valuations_hsv = classify_by_color_and_size(self.detected_circles, self.original_bgr)
        annotated_hsv = draw_circles(self.original_bgr, self.detected_circles)
        self.annotated_bgr_hsv = self._draw_valuations(annotated_hsv, self.valuations_hsv)
        _, libelle_hsv = valeur_totale(self.valuations_hsv)
        self.total_hsv_var.set(f"Total principal : {libelle_hsv}" if self.valuations_hsv else "")
        self._set_details(self.details_hsv, self._format_valuations_hsv(self.valuations_hsv))

        # Classification HLS
        self.valuations_hls = classify_hls(self.detected_circles, self.original_bgr)
        annotated_hls = draw_circles(self.original_bgr, self.detected_circles)
        self.annotated_bgr_hls = self._draw_valuations_hls(annotated_hls, self.valuations_hls)
        _, libelle_hls = valeur_totale_hls(self.valuations_hls)
        self.total_hls_var.set(f"Total HLS : {libelle_hls}" if self.valuations_hls else "")
        self._set_details(self.details_hls, self._format_valuations_hls(self.valuations_hls))

        # Classification Filtres
        self.valuations_filtres = classify_filtres(self.detected_circles, self.original_bgr)
        annotated_filtres = draw_circles(self.original_bgr, self.detected_circles)
        self.annotated_bgr_filtres = self._draw_valuations_filtres(annotated_filtres, self.valuations_filtres)
        _, libelle_filtres = valeur_totale_filtres(self.valuations_filtres)
        self.total_filtres_var.set(f"Total Filtres : {libelle_filtres}" if self.valuations_filtres else "")
        self._set_details(self.details_filtres, self._format_valuations_filtres(self.valuations_filtres))

        # Classification Combinée
        self.valuations_combine = classify_combine(self.detected_circles, self.original_bgr)
        annotated_combine = draw_circles(self.original_bgr, self.detected_circles)
        self.annotated_bgr_combine = self._draw_valuations_combine(annotated_combine, self.valuations_combine)
        _, libelle_combine = valeur_totale_combinee(self.valuations_combine)
        self.total_combine_var.set(f"Total Combiné : {libelle_combine}" if self.valuations_combine else "")
        self._set_details(self.details_combine, self._format_valuations_combine(self.valuations_combine))

        self.status_var.set("Analyse terminee.")
        # Afficher l'image correspondant a l'onglet actif
        self._show_current_tab_image()

    def show_original(self) -> None:
        if self.original_bgr is None:
            return
        self.status_var.set("Affichage de l'image originale.")
        self._show_bgr_image(self.original_bgr)

    def show_labeling(self) -> None:
        """Affiche les annotations ground truth (LabelMe) sur l'image."""
        if self.original_bgr is None or self.current_path is None:
            messagebox.showinfo("Aucune image", "Charge d'abord une image.")
            return

        # Chercher le JSON d'annotation correspondant
        stem = self.current_path.stem
        json_path = BDD_DIR / f"{stem}.json"
        if not json_path.exists():
            messagebox.showinfo("Pas d'annotation", f"Aucun fichier d'annotation pour {stem}")
            return

        annotation = load_labelme_annotation(json_path)
        circles = annotation["circles"]

        if not circles:
            messagebox.showinfo("Annotation vide", "Aucun cercle annote dans ce fichier.")
            return

        # Mise a l'echelle
        img_h, img_w = self.original_bgr.shape[:2]
        ann_w = annotation["image_width"]
        ann_h = annotation["image_height"]
        scale_x = img_w / ann_w if ann_w else 1.0
        scale_y = img_h / ann_h if ann_h else 1.0

        canvas = self.original_bgr.copy()
        for c in circles:
            color = LABEL_COLORS.get(c.label, (255, 255, 255))
            cx = int(c.x * scale_x)
            cy = int(c.y * scale_y)
            radius = int(c.radius * ((scale_x + scale_y) / 2))
            cv2.circle(canvas, (cx, cy), radius, color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.4, radius / 45)
            thickness = max(1, int(scale * 1.5))
            (tw, th), _ = cv2.getTextSize(c.label, font, scale, thickness)
            tx, ty = cx - tw // 2, cy - radius - 10
            cv2.putText(canvas, c.label, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, c.label, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

        self.status_var.set(f"Labeling ground truth : {len(circles)} piece(s) annotee(s)")
        self._show_bgr_image(canvas)

    def _on_tab_change(self, _event=None) -> None:
        self._show_current_tab_image()

    def _show_current_tab_image(self) -> None:
        tab_idx = self.notebook.index(self.notebook.select())
        if tab_idx == 0 and self.annotated_bgr_hsv is not None:
            self._show_bgr_image(self.annotated_bgr_hsv)
        elif tab_idx == 1 and self.annotated_bgr_hls is not None:
            self._show_bgr_image(self.annotated_bgr_hls)
        elif tab_idx == 2 and self.annotated_bgr_filtres is not None:
            self._show_bgr_image(self.annotated_bgr_filtres)
        elif tab_idx == 3 and self.annotated_bgr_combine is not None:
            self._show_bgr_image(self.annotated_bgr_combine)
        elif self.original_bgr is not None:
            self._show_bgr_image(self.original_bgr)

    def _refresh_canvas(self, _event: tk.Event) -> None:
        self._show_current_tab_image()

    def _show_bgr_image(self, image_bgr) -> None:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        canvas_width = max(self.canvas.winfo_width(), 200)
        canvas_height = max(self.canvas.winfo_height(), 200)
        image.thumbnail((canvas_width - 16, canvas_height - 16), Image.Resampling.LANCZOS)

        self._canvas_photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=self._canvas_photo, anchor="center")

    def _set_details(self, widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _draw_valuations(self, image, valuations: list[ValeurPiece]):
        canvas = image.copy()
        for v in valuations:
            cx, cy, r = v.cercle.x, v.cercle.y, v.cercle.radius
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = max(0.45, r / 38)
            thickness = max(1, int(scale * 1.8))
            text = v.denomination
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx, ty = cx - tw // 2, cy + th // 2
            cv2.putText(canvas, text, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(canvas, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return canvas

    def _draw_valuations_hls(self, image, valuations: list[ValeurPieceHLS]):
        canvas = image.copy()
        for v in valuations:
            cx, cy, r = v.cercle.x, v.cercle.y, v.cercle.radius
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = max(0.45, r / 38)
            thickness = max(1, int(scale * 1.8))
            text = v.denomination
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx, ty = cx - tw // 2, cy + th // 2
            cv2.putText(canvas, text, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            # Couleur cyan pour distinguer visuellement du HSV
            cv2.putText(canvas, text, (tx, ty), font, scale, (255, 255, 0), thickness, cv2.LINE_AA)
        return canvas

    def _format_valuations_hsv(self, valuations: list[ValeurPiece]) -> str:
        if not valuations:
            return "Aucune piece detectee sur cette image."
        ICONE_GROUPE = {"cuivre": "[Cu]", "or": "[Au]", "bimetallic": "[Bi]"}
        lines = []
        for i, v in enumerate(valuations, start=1):
            icone = ICONE_GROUPE.get(v.groupe_couleur, "[?]")
            conf = int(v.confiance * 100)
            lines.append(f"{i:02d}  {v.denomination:<4s}  {icone}  conf={conf:3d}%  r={v.cercle.radius}px")
        return "\n".join(lines)

    def _format_valuations_hls(self, valuations: list[ValeurPieceHLS]) -> str:
        if not valuations:
            return "Aucune piece detectee sur cette image."
        ICONE_GROUPE = {"cuivre": "[Cu]", "or": "[Au]", "bimetallic": "[Bi]"}
        lines = []
        for i, v in enumerate(valuations, start=1):
            icone = ICONE_GROUPE.get(v.groupe_couleur, "[?]")
            conf = int(v.confiance * 100)
            lines.append(f"{i:02d}  {v.denomination:<4s}  {icone}  conf={conf:3d}%  r={v.cercle.radius}px")
        return "\n".join(lines)

    def _draw_valuations_filtres(self, image, valuations: list[ValeurPieceFiltre]):
        canvas = image.copy()
        for v in valuations:
            cx, cy, r = v.cercle.x, v.cercle.y, v.cercle.radius
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = max(0.45, r / 38)
            thickness = max(1, int(scale * 1.8))
            text = v.denomination
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx, ty = cx - tw // 2, cy + th // 2
            cv2.putText(canvas, text, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            # Couleur orange pour distinguer des autres méthodes
            cv2.putText(canvas, text, (tx, ty), font, scale, (0, 165, 255), thickness, cv2.LINE_AA)
        return canvas

    def _format_valuations_filtres(self, valuations: list[ValeurPieceFiltre]) -> str:
        if not valuations:
            return "Aucune piece detectee sur cette image."
        ICONE_GROUPE = {"cuivre": "[Cu]", "or": "[Au]", "bimetallic": "[Bi]"}
        lines = []
        for i, v in enumerate(valuations, start=1):
            icone = ICONE_GROUPE.get(v.groupe_couleur, "[?]")
            conf = int(v.confiance * 100)
            lines.append(f"{i:02d}  {v.denomination:<4s}  {icone}  conf={conf:3d}%  r={v.cercle.radius}px")
        return "\n".join(lines)

    def _draw_valuations_combine(self, image, valuations: list[ValeurPieceCombinee]):
        canvas = image.copy()
        for v in valuations:
            cx, cy, r = v.cercle.x, v.cercle.y, v.cercle.radius
            font = cv2.FONT_HERSHEY_DUPLEX
            scale = max(0.45, r / 38)
            thickness = max(1, int(scale * 1.8))
            text = v.denomination
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx, ty = cx - tw // 2, cy + th // 2
            cv2.putText(canvas, text, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            # Vert vif pour le combiné
            cv2.putText(canvas, text, (tx, ty), font, scale, (0, 255, 100), thickness, cv2.LINE_AA)
        return canvas

    def _format_valuations_combine(self, valuations: list[ValeurPieceCombinee]) -> str:
        if not valuations:
            return "Aucune piece detectee sur cette image."
        ICONE_GROUPE = {"cuivre": "[Cu]", "or": "[Au]", "bimetallic": "[Bi]"}
        lines = []
        for i, v in enumerate(valuations, start=1):
            icone = ICONE_GROUPE.get(v.groupe_couleur, "[?]")
            conf = int(v.confiance * 100)
            lines.append(
                f"{i:02d}  {v.denomination:<4s}  {icone}  conf={conf:3d}%  "
                f"r={v.cercle.radius}px  [{v.methode_dominante}]"
            )
        return "\n".join(lines)


def main() -> None:
    root = tk.Tk()
    app = EuroVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
