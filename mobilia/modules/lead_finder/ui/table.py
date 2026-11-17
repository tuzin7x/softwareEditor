"""Tabela de leads: modelo paginado (os dados vêm do banco, página a página) e botão OPEN WHATSAPP por linha."""
from __future__ import annotations

from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QStyle, QStyledItemDelegate

from mobilia.ui import theme

from .. import phone as ph
from ..models import DO_NOT_CONTACT, PRIORITY_LABEL, STATUS_LABEL, WA_CONFIRMED, WA_LABEL
from ..sources import TITLES
from ..sources.brasilapi import format_cnpj
from . import icons, style

COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("name", "Empresa", 230), ("city", "Cidade / UF", 150), ("address", "Endereço comercial", 220),
    ("phone", "Telefone", 130), ("whatsapp", "WhatsApp", 168), ("website", "Site oficial", 170),
    ("cnpj", "CNPJ", 150), ("category", "Categoria", 150), ("source", "Fonte · consulta", 190),
    ("priority", "Prioridade", 110), ("status", "Status", 160),
)
COL = {k: i for i, (k, _, _) in enumerate(COLUMNS)}
WA_TEXT = "OPEN WHATSAPP"


def short_date(iso: str) -> str:
    return f"{iso[8:10]}/{iso[5:7]}/{iso[:4]}" if iso and len(iso) >= 10 else ""


def sources_text(row: dict) -> str:
    names = [TITLES.get(s, s) for s in (row.get("sources") or "").split(",") if s]
    return " + ".join(names) + (f" · {short_date(row.get('last_fetched', ''))}" if row.get("last_fetched") else "")


def can_whatsapp(row: dict) -> bool:
    return bool(row.get("wa_number")) and row.get("status") != DO_NOT_CONTACT


class LeadTableModel(QAbstractTableModel):
    def __init__(self):
        super().__init__()
        self.rows: list[dict] = []
        self.total = 0

    def set_rows(self, rows: list[dict], total: int) -> None:
        self.beginResetModel()
        self.rows, self.total = rows, total
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):  # noqa: N802
        return 0 if parent.isValid() else len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):  # noqa: N802
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLUMNS[section][1].upper()
        return None

    def lead_id(self, row: int) -> int | None:
        return self.rows[row]["id"] if 0 <= row < len(self.rows) else None

    def row_dict(self, row: int) -> dict | None:
        return self.rows[row] if 0 <= row < len(self.rows) else None

    def text(self, r: dict, key: str) -> str:
        if key == "name":
            return r["name"]
        if key == "city":
            return " / ".join(x for x in (r["city"], r["uf"]) if x)
        if key == "address":
            return r["address"]
        if key == "phone":
            info = ph.normalize(r["phone_e164"] or r["phone"])
            return info.national if info.valid else (r["phone"] or "")
        if key == "whatsapp":
            return WA_LABEL.get(r["wa_status"], "")
        if key == "website":
            return r["domain"]
        if key == "cnpj":
            return format_cnpj(r["cnpj"]) if r["cnpj"] else ""
        if key == "category":
            return r["category"]
        if key == "source":
            return sources_text(r)
        if key == "priority":
            p = max(r["priority"] or 0, r["priority_user"] or 0)
            return PRIORITY_LABEL[p] + (f" · {r['score']}" if r["score"] is not None else "")
        if key == "status":
            return STATUS_LABEL.get(r["status"], r["status"])
        return ""

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        r = self.rows[index.row()]
        key = COLUMNS[index.column()][0]
        if role == Qt.DisplayRole:
            return "" if key in ("whatsapp", "status") else self.text(r, key)
        if role == Qt.UserRole:
            return r["id"]
        if role == Qt.UserRole + 1:
            return r
        if role == Qt.ToolTipRole:
            if key == "whatsapp":
                if r["status"] == DO_NOT_CONTACT:
                    return "Esta empresa pediu para não ser contatada."
                if not r["wa_number"]:
                    return "Nenhum número de WhatsApp conhecido."
                info = ph.normalize(r["wa_number"])
                tip = f"{info.national} — {WA_LABEL.get(r['wa_status'], '')}"
                if r["wa_status"] != WA_CONFIRMED:
                    tip += "\nCelular não garante WhatsApp: confirme ao conversar."
                return tip
            if key == "source":
                return sources_text(r) + ("\nInclui dados do Google: guardados por até 30 dias e não exportáveis."
                                          if r["restricted"] else "")
            if key == "name" and r["legal_name"] and r["legal_name"] != r["name"]:
                return r["legal_name"]
            return self.text(r, key) or None
        if role == Qt.ForegroundRole:
            if key in ("address", "category", "source", "cnpj"):
                return QColor(theme.TEXT_2)
            if key == "priority":
                return QColor(style.PRIORITY_COLOR[max(r["priority"] or 0, r["priority_user"] or 0)])
            if r["status"] == DO_NOT_CONTACT and key == "name":
                return QColor(theme.MUTED)
        if role == Qt.FontRole and key == "name":
            f = QFont(); f.setWeight(QFont.DemiBold)
            return f
        return None


class PillDelegate(QStyledItemDelegate):
    """Status do CRM como etiqueta colorida."""

    def paint(self, p: QPainter, opt, index):
        r = index.data(Qt.UserRole + 1)
        if r is None:
            return super().paint(p, opt, index)
        self._background(p, opt)
        text = STATUS_LABEL.get(r["status"], r["status"])
        color = QColor(style.STATUS_COLOR.get(r["status"], theme.MUTED))
        p.save(); p.setRenderHint(QPainter.Antialiasing)
        f = QFont(opt.font); f.setPixelSize(11); f.setWeight(QFont.DemiBold); p.setFont(f)
        w = p.fontMetrics().horizontalAdvance(text) + 22
        rect = QRectF(opt.rect.left() + 8, opt.rect.center().y() - 11, min(w, opt.rect.width() - 16), 22)
        bg = QColor(color); bg.setAlphaF(0.13)
        p.setPen(QPen(QColor(color.red(), color.green(), color.blue(), 90), 1)); p.setBrush(bg)
        p.drawRoundedRect(rect, 11, 11)
        p.setPen(color)
        p.drawText(rect, Qt.AlignCenter, p.fontMetrics().elidedText(text, Qt.ElideRight, int(rect.width()) - 10))
        p.restore()

    @staticmethod
    def _background(p, opt):
        if opt.state & QStyle.State_Selected:
            p.fillRect(opt.rect, QColor(theme.SELECTED_BG))


class WhatsAppDelegate(PillDelegate):
    """Botão OPEN WHATSAPP desenhado na célula; o clique emite ``clicked(lead_id)``."""

    clicked = Signal(int)

    def button_rect(self, rect: QRect) -> QRectF:
        return QRectF(rect.left() + 8, rect.center().y() - 13, min(150, rect.width() - 16), 26)

    def paint(self, p: QPainter, opt, index):
        r = index.data(Qt.UserRole + 1)
        if r is None:
            return QStyledItemDelegate.paint(self, p, opt, index)
        self._background(p, opt)
        p.save(); p.setRenderHint(QPainter.Antialiasing)
        f = QFont(opt.font); f.setPixelSize(11); f.setWeight(QFont.DemiBold)
        f.setLetterSpacing(QFont.AbsoluteSpacing, 0.6); p.setFont(f)
        if can_whatsapp(r):
            rect = self.button_rect(opt.rect)
            col = QColor(style.WA_COLOR)
            hover = bool(opt.state & QStyle.State_MouseOver)
            bg = QColor(col); bg.setAlphaF(0.22 if hover else 0.11)
            border = QColor(col); border.setAlphaF(0.9 if r["wa_status"] == WA_CONFIRMED else 0.45)
            pen = QPen(border, 1)
            if r["wa_status"] != WA_CONFIRMED:
                pen.setStyle(Qt.DashLine)
            p.setPen(pen); p.setBrush(bg); p.drawRoundedRect(rect, 8, 8)
            p.drawPixmap(int(rect.left() + 9), int(rect.center().y() - 7), icons.pixmap("whatsapp", 14, style.WA_COLOR))
            p.setPen(col)
            p.drawText(rect.adjusted(26, 0, -6, 0), Qt.AlignVCenter | Qt.AlignLeft, WA_TEXT)
        else:
            p.setPen(QColor(theme.DANGER if r["status"] == DO_NOT_CONTACT else theme.MUTED))
            f.setLetterSpacing(QFont.AbsoluteSpacing, 0); f.setWeight(QFont.Normal); p.setFont(f)
            txt = "Não contatar" if r["status"] == DO_NOT_CONTACT else "—"
            p.drawText(opt.rect.adjusted(12, 0, -4, 0), Qt.AlignVCenter | Qt.AlignLeft, txt)
        p.restore()

    def editorEvent(self, event, model, opt, index):  # noqa: N802
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            r = index.data(Qt.UserRole + 1)
            if r is not None and can_whatsapp(r) and self.button_rect(opt.rect).contains(event.position()):
                self.clicked.emit(r["id"])
                return True
        return super().editorEvent(event, model, opt, index)

    def sizeHint(self, opt, index):  # noqa: N802
        return QSize(160, 44)
