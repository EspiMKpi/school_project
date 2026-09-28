# -*- coding: utf-8 -*-
"""
Full Document Generator for PTTK Assignment (Subject No. 09)
Student: Nguyen Dai Dung (D23DCVT103 - Digit 3)
Outputs:
- 09D23DCVT103.md
- 09D23DCVT103.docx
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

sys.stdout.reconfigure(encoding='utf-8')

# Import plantuml & metadata
from generate_full_report import (
    PUML_UC_OVERALL, PUML_UC_MOD1, PUML_UC_MOD2,
    PUML_ENTITY_SYSTEM, PUML_CLASS_MOD1, PUML_CLASS_MOD2,
    PUML_STATE_MOD1, PUML_STATE_MOD2, PUML_COMM_MOD1, PUML_COMM_MOD2,
    GLOSSARY_LIST, OVERALL_UC_DESC, MOD1_UC_DESC, MOD2_UC_DESC
)

# -------------------------------------------------------------
# HELPER FUNCTIONS FOR DOCX STYLING
# -------------------------------------------------------------
def set_cell_background(cell, fill_hex):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_table_borders(table, color="D3D3D3", sz="4", val="single"):
    tblPr = table._tbl.tblPr
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'  <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'  <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>'
        f'</w:tblBorders>'
    )
    tblPr.append(borders)

def add_heading_with_spacing(doc, text, level):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    if level == 1:
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        for run in h.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(16)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    elif level == 2:
        h.paragraph_format.space_before = Pt(12)
        h.paragraph_format.space_after = Pt(4)
        for run in h.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(13)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x2E, 0x75, 0xB6)
    elif level == 3:
        h.paragraph_format.space_before = Pt(8)
        h.paragraph_format.space_after = Pt(3)
        for run in h.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(11.5)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return h

def add_paragraph_styled(doc, text, bold_prefix=None, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.bold = True
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(11)
        r_pre.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(11)
    r.italic = italic
    r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
    return p

def add_code_box(doc, code_str, title=None):
    if title:
        p_t = doc.add_paragraph()
        p_t.paragraph_format.space_before = Pt(6)
        p_t.paragraph_format.space_after = Pt(2)
        r_t = p_t.add_run(f"Mã nguồn PlantUML: {title}")
        r_t.bold = True
        r_t.font.name = 'Times New Roman'
        r_t.font.size = Pt(10)
        r_t.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F4F6F8")
    set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
    
    # Border for code box
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'  <w:top w:val="single" w:sz="6" w:space="0" w:color="BDC3C7"/>'
        f'  <w:left w:val="single" w:sz="18" w:space="0" w:color="2E75B6"/>'
        f'  <w:bottom w:val="single" w:sz="6" w:space="0" w:color="BDC3C7"/>'
        f'  <w:right w:val="single" w:sz="6" w:space="0" w:color="BDC3C7"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(code_str.strip())
    r.font.name = 'Consolas'
    r.font.size = Pt(8.5)
    r.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(6)

def add_table_styled(doc, headers, data, col_widths=None):
    table = doc.add_table(rows=len(data) + 1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table)

    # Format header row
    hdr_cells = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr_cells[i].text = h
        set_cell_background(hdr_cells[i], "1F4E79")
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=120, right=120)
        p = hdr_cells[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        for run in p.runs:
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    # Format data rows
    for r_idx, row_data in enumerate(data):
        row_cells = table.rows[r_idx + 1].cells
        bg_color = "F9FAFB" if (r_idx % 2 == 1) else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_color)
            set_cell_margins(row_cells[c_idx], top=80, bottom=80, left=100, right=100)
            p = row_cells[c_idx].paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            for run in p.runs:
                run.font.name = 'Times New Roman'
                run.font.size = Pt(9.5)
                run.font.color.rgb = RGBColor(0x33, 0x33, 0x33)

    if col_widths:
        for row in table.rows:
            for idx, w in enumerate(col_widths):
                row.cells[idx].width = Inches(w)

    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_after = Pt(6)
    return table


# -------------------------------------------------------------
# BUILD DOCX DOCUMENT
# -------------------------------------------------------------
def build_docx():
    print("Building 09D23DCVT103.docx...")
    doc = docx.Document()

    # Set page margins to 1 inch
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # COVER / HEADER BLOCK
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("HỌC VIỆN CÔNG NGHỆ BƯU CHÍNH VIỄN THÔNG\nKHOA CÔNG NGHỆ THÔNG TIN")
    r_inst.bold = True
    r_inst.font.name = 'Times New Roman'
    r_inst.font.size = Pt(13)
    r_inst.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(16)
    p_title.paragraph_format.space_after = Pt(6)
    r_title = p_title.add_run("BÁO CÁO BÀI TẬP LỚN\nPHÂN TÍCH VÀ THIẾT KẾ HỆ THỐNG THÔNG TIN (PTTK)")
    r_title.bold = True
    r_title.font.name = 'Times New Roman'
    r_title.font.size = Pt(18)
    r_title.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(14)
    r_sub = p_sub.add_run("ĐỀ TÀI SỐ 09: ONLINE SUPERMARKET MANAGEMENT SYSTEM\n(HỆ THỐNG QUẢN LÝ SIÊU THỊ TRỰC TUYẾN)")
    r_sub.bold = True
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(13)
    r_sub.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)

    # Student Info Table
    info_table = doc.add_table(rows=4, cols=2)
    info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(info_table, color="B0C4DE", sz="4")
    infos = [
        ("Sinh viên thực hiện:", "NGUYỄN ĐẠI DŨNG"),
        ("Mã sinh viên:", "D23DCVT103 (Số đuôi quy định: 3)"),
        ("Lớp học phần:", "D23CQVT01-B - Phân tích thiết kế hệ thống"),
        ("Quy ước đặt tên đề bài:", "Tên Use Case: Tiếng Anh + 3 (SearchItem 3); Tên Class: Tiếng Anh + 3 (Item 3)")
    ]
    for idx, (label, val) in enumerate(infos):
        c0 = info_table.rows[idx].cells[0]
        c1 = info_table.rows[idx].cells[1]
        c0.text = label
        c1.text = val
        set_cell_background(c0, "F0F4F8")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)
        c0.paragraphs[0].runs[0].bold = True
        c0.paragraphs[0].runs[0].font.name = 'Times New Roman'
        c0.paragraphs[0].runs[0].font.size = Pt(10)
        c1.paragraphs[0].runs[0].font.name = 'Times New Roman'
        c1.paragraphs[0].runs[0].font.size = Pt(10)
    info_table.rows[0].cells[0].width = Inches(2.2)
    info_table.rows[0].cells[1].width = Inches(4.3)

    doc.add_page_break()

    # ------------------ ASSIGNMENT 1 ------------------
    add_heading_with_spacing(doc, "ASSIGNMENT 1: THU THẬP YÊU CẦU & MÔ HÌNH HÓA USE CASE", level=1)
    
    add_heading_with_spacing(doc, "1.1. Lập bảng từ khóa (Glossary List) theo mẫu chuẩn", level=2)
    add_paragraph_styled(doc, "Quá trình khảo sát và thu thập yêu cầu chuyên môn được tiến hành theo đúng 3 bước tại Mục 3.1.1 của giáo trình: Liệt kê từ khóa (brainstorming), Phân nhóm từ khóa, và Lập bảng giải thích ngữ nghĩa chuyên môn chi tiết.")

    add_heading_with_spacing(doc, "Bước 1 & 2: Phân loại từ khóa theo 3 nhóm chuyên môn", level=3)
    kw_headers = ["Nhóm 1: Con người (Actors)", "Nhóm 2: Hoạt động (Activities)", "Nhóm 3: Vật, đối tượng (Entities/Data)"]
    kw_data = [
        (
            "- User 3 (Người dùng)\n- Member 3 (Thành viên)\n- Customer 3 (Khách hàng)\n- Staff 3 (Nhân viên)\n- WarehouseStaff 3 (Thủ kho)\n- Saler 3 (NV bán hàng)\n- ManagementStaff 3 (NV quản lý)\n- DeliveryStaff 3 (NV giao hàng)\n- Supplier 3 (Nhà cung cấp)",
            "- Login 3 (Đăng nhập)\n- Logout 3 (Đăng xuất)\n- ChangePassword 3 (Đổi mật khẩu)\n- RegisterMember 3 (Đăng ký)\n- SearchItem 3 (Tìm kiếm mặt hàng)\n- ViewItemDetail 3 (Xem chi tiết)\n- OrderOnline 3 (Đặt hàng online)\n- BuyAtCounter 3 (Mua tại quầy)\n- SellAtCounter 3 (Bán tại quầy)\n- CreateCounterInvoice 3 (Lập HĐ)\n- ImportGoods 3 (Nhập hàng NCC)\n- ManageItem 3 (Quản lý mặt hàng)\n- ManageSupplier 3 (Quản lý NCC)\n- BrowseAndExportOrder 3 (Duyệt xuất đơn)\n- PrintInvoice 3 (In hóa đơn)\n- DeliverOrder 3 (Giao hàng)\n- ViewStatistics 3 (Xem thống kê)",
            "- Supermarket 3 (Siêu thị)\n- Warehouse 3 (Kho hàng)\n- Item 3 (Mặt hàng / Hàng hóa)\n- Category 3 (Danh mục)\n- Unit 3 (Đơn vị tính)\n- Price 3 (Đơn giá)\n- StockQuantity 3 (Số lượng tồn)\n- Description 3 (Mô tả)\n- Supplier 3 (Nhà cung cấp)\n- ImportSlip 3 (Phiếu nhập kho)\n- ImportSlipDetail 3 (CT phiếu nhập)\n- Order 3 (Đơn đặt hàng)\n- OrderStatus 3 (Trạng thái đơn)\n- OrderItem 3 (Chi tiết đơn hàng)\n- Invoice 3 (Hóa đơn)\n- ItemStatistic 3 (TK mặt hàng)\n- SupplierStatistic 3 (TK NCC)\n- RevenueStatistic 3 (TK doanh thu)\n- FullName 3 (Họ tên)\n- Address 3 (Địa chỉ)"
        )
    ]
    add_table_styled(doc, kw_headers, kw_data, col_widths=[2.1, 2.3, 2.1])

    add_heading_with_spacing(doc, "Bước 3: Bảng giải thích thuật ngữ chuyên môn (Glossary Table)", level=3)
    add_paragraph_styled(doc, "Bảng giải thích thuật ngữ ngữ nghĩa chi tiết cho toàn bộ các khái niệm nòng cốt trong hệ thống siêu thị trực tuyến:")
    gl_headers = ["TT", "Tên Tiếng Việt", "Tên Tiếng Anh (Kèm số 3)", "Giải thích ngữ nghĩa nghiệp vụ chi tiết"]
    add_table_styled(doc, gl_headers, GLOSSARY_LIST, col_widths=[0.4, 1.4, 1.5, 3.2])

    add_heading_with_spacing(doc, "1.2. Mô tả hệ thống bằng ngôn ngữ tự nhiên (5 bước chuẩn case study)", level=2)
    add_paragraph_styled(doc, "Thực hiện đầy đủ và nghiêm ngặt 5 bước theo phương pháp tại Mục 3.1.2 của giáo trình:")

    add_heading_with_spacing(doc, "Bước 1: Giới thiệu mục đích hệ thống", level=3)
    add_paragraph_styled(doc, "Hệ thống trang web quản lý siêu thị trực tuyến (Online Supermarket Management System) được xây dựng nhằm cung cấp giải pháp chuyển đổi số bán lẻ toàn diện: hỗ trợ khách hàng tìm kiếm và đặt mua hàng trực tuyến hoặc mua hàng trực tiếp tại quầy; nhân viên bán hàng tính tiền và xuất hóa đơn nhanh chóng; thủ kho quản lý kho hàng, nhập hàng từ nhà cung cấp, duyệt và xuất đơn hàng trực tuyến bàn giao cho nhân viên giao hàng; ban quản lý theo dõi sát sao tình hình kinh doanh thông qua các báo cáo thống kê chuyên sâu về mặt hàng, nhà cung cấp và doanh thu.")

    add_heading_with_spacing(doc, "Bước 2: Phạm vi hệ thống (Actors và phân quyền chức năng)", level=3)
    add_paragraph_styled(doc, "• Thành viên hệ thống (Member 3 / User 3): Đăng nhập (Login 3), Đăng xuất (Logout 3), Đổi mật khẩu (ChangePassword 3), Cập nhật thông tin cá nhân (UpdateProfile 3).")
    add_paragraph_styled(doc, "• Khách hàng (Customer 3): Đăng ký thành viên (RegisterMember 3), Tìm kiếm mặt hàng theo từ khóa (SearchItem 3), Xem chi tiết mặt hàng (ViewItemDetail 3), Đặt hàng online (OrderOnline 3), Mua hàng trực tiếp tại quầy (BuyAtCounter 3).")
    add_paragraph_styled(doc, "• Nhân viên bán hàng (Saler 3): Bán hàng tại quầy cho khách (SellAtCounter 3), Lập hóa đơn bán lẻ trực tiếp (CreateCounterInvoice 3).")
    add_paragraph_styled(doc, "• Nhân viên thủ kho (WarehouseStaff 3): Nhập hàng từ nhà cung cấp (ImportGoods 3), Quản lý mặt hàng (ManageItem 3: thêm, sửa, xóa), Quản lý nhà cung cấp (ManageSupplier 3: thêm, sửa, xóa), Duyệt đơn hàng online, cập nhật xuất kho và bàn giao cho NV giao hàng (BrowseAndExportOrder 3), In hóa đơn bán hàng (PrintInvoice 3).")
    add_paragraph_styled(doc, "• Nhân viên giao hàng (DeliveryStaff 3): Nhận kiện hàng và hóa đơn từ thủ kho, giao hàng đến tận tay người mua và cập nhật kết quả giao (DeliverOrder 3).")
    add_paragraph_styled(doc, "• Nhân viên quản lý (ManagementStaff 3): Thiết lập thông tin tham số chung (ManageGeneralInfo 3), Xem các loại thống kê báo cáo (ViewStatistics 3) gồm Thống kê mặt hàng (ViewItemStats 3), Thống kê nhà cung cấp (ViewSupplierStats 3), Thống kê doanh thu (ViewRevenueStats 3).")

    add_heading_with_spacing(doc, "Bước 3: Hoạt động nghiệp vụ của các chức năng (Mô tả chi tiết)", level=3)
    add_paragraph_styled(doc, "1. Module 1: Customer searches for items (SearchItem 3)", bold_prefix="• ")
    add_paragraph_styled(doc, "Khách hàng truy cập hệ thống -> chọn menu tìm kiếm mặt hàng -> nhập từ khóa tên mặt hàng cần tìm vào ô tìm kiếm -> hệ thống tiến hành truy vấn cơ sở dữ liệu và hiển thị danh sách các mặt hàng có tên chứa từ khóa (mã mặt hàng, tên mặt hàng, đơn vị tính, đơn giá, hình ảnh đại diện, số lượng còn trong kho) -> Khách hàng click vào một mặt hàng trong danh sách để xem chi tiết -> Hệ thống hiển thị toàn bộ thông tin chi tiết của mặt hàng (mã, tên, đơn giá, đơn vị tính, số lượng tồn kho, nhà sản xuất, hạn sử dụng, thông số kỹ thuật, mô tả chi tiết sản phẩm).")
    
    add_paragraph_styled(doc, "2. Module 2: Warehouse staff approves orders + export (BrowseAndExportOrder 3)", bold_prefix="• ")
    add_paragraph_styled(doc, "Thủ kho đăng nhập vào hệ thống -> chọn menu duyệt đơn hàng trực tuyến -> hệ thống truy xuất và hiển thị danh sách các đơn đặt hàng online đang ở trạng thái chưa xuất kho (mã đơn hàng, ngày đặt, tên khách hàng, địa chỉ nhận hàng, số điện thoại, tổng tiền, trạng thái) -> Thủ kho chọn một đơn hàng chưa xuất từ danh sách để kiểm tra -> Hệ thống hiển thị chi tiết các mặt hàng trong đơn, số lượng đặt, vị trí lưu kho tương ứng và danh sách các nhân viên giao hàng đang sẵn sàng nhận việc -> Thủ kho chọn nhân viên giao hàng phụ trách -> Thủ kho bấm xác nhận xuất kho -> Hệ thống cập nhật trạng thái đơn hàng sang 'Đã xuất kho' (hoặc 'Đang giao'), đồng thời tự động khởi tạo hóa đơn bán hàng hợp lệ -> Hệ thống thực hiện lệnh in hóa đơn bán hàng ra máy in -> Thủ kho lấy hàng hóa thực tế từ kho, đóng gói và bàn giao toàn bộ hàng hóa kèm theo hóa đơn đã in cho nhân viên giao hàng mang đi giao.")

    add_paragraph_styled(doc, "3. Chức năng Nhập hàng từ nhà cung cấp (ImportGoods 3):", bold_prefix="• ")
    add_paragraph_styled(doc, "Thủ kho đăng nhập -> chọn chức năng nhập hàng -> chọn nhà cung cấp -> lập phiếu nhập hàng -> nhập danh sách mặt hàng, số lượng và giá nhập -> lưu phiếu nhập -> hệ thống cập nhật tăng tồn kho.")

    add_paragraph_styled(doc, "4. Chức năng Quản lý mặt hàng và Nhà cung cấp (ManageItem 3, ManageSupplier 3):", bold_prefix="• ")
    add_paragraph_styled(doc, "Thủ kho tra cứu danh sách, thực hiện thêm mới, chỉnh sửa thông tin hoặc vô hiệu hóa mặt hàng/nhà cung cấp khi có thay đổi.")

    add_paragraph_styled(doc, "5. Chức năng Bán hàng tại quầy (SellAtCounter 3):", bold_prefix="• ")
    add_paragraph_styled(doc, "Thu ngân quét mã mặt hàng khách mua tại quầy -> hệ thống tính tiền -> thanh toán và in hóa đơn -> trừ tồn kho.")

    add_paragraph_styled(doc, "6. Chức năng Xem thống kê báo cáo (ViewStatistics 3):", bold_prefix="• ")
    add_paragraph_styled(doc, "Quản lý chọn loại thống kê (mặt hàng, nhà cung cấp, doanh thu) -> chọn khoảng thời gian -> hệ thống tính toán và hiển thị bảng biểu số liệu tổng hợp cùng đồ thị trực quan.")

    add_heading_with_spacing(doc, "Bước 4: Thông tin các đối tượng cần xử lý, quản lý", level=3)
    add_paragraph_styled(doc, "- Thông tin con người: User 3 (username, password, email, phone, role, note); FullName 3 (firstName, middleName, lastName); Address 3 (houseNumber, street, ward, district, city); Customer 3 (customerCode, membershipLevel, rewardPoints); Staff 3 (staffCode, position, salaryRate); WarehouseStaff 3 (warehouseArea); Saler 3 (counterNumber); ManagementStaff 3 (department); DeliveryStaff 3 (vehicleNumber, deliveryArea, deliveryStatus); Supplier 3 (code, name, taxCode, phone, email, bankAccount, note).")
    add_paragraph_styled(doc, "- Thông tin cơ sở vật chất: Supermarket 3 (code, name, address, phone); Warehouse 3 (code, name, location, capacity); CheckoutCounter 3 (counterNumber, location, status).")
    add_paragraph_styled(doc, "- Thông tin chuyên môn, vận hành: Category 3 (code, name, description); Item 3 (code, name, unit, price, description, stockQuantity, origin, expiryDate); ImportSlip 3 (code, importDate, totalAmount, note); ImportSlipDetail 3 (quantity, importPrice, amount); Order 3 (code, orderDate, status, totalAmount, deliveryAddress, recipientPhone, note); OrderItem 3 (quantity, unitPrice, discount, amount); Invoice 3 (code, createdDate, paymentMethod, totalAmount, paymentStatus, note).")
    add_paragraph_styled(doc, "- Thông tin thống kê: Statistic 3 (startDate, endDate, description); ItemStatistic 3 (totalQuantitySold, totalRevenue); SupplierStatistic 3 (totalImportAmount, totalImportSlips); RevenueStatistic 3 (totalOrders, totalRevenue, netProfit).")

    add_heading_with_spacing(doc, "Bước 5: Quan hệ số lượng giữa các đối tượng", level=3)
    add_paragraph_styled(doc, "• User 3 - FullName 3: 1 - 1; User 3 - Address 3: 1 - 1; Supplier 3 - Address 3: 1 - 1.")
    add_paragraph_styled(doc, "• Category 3 - Item 3: 1 - n (Một danh mục có nhiều mặt hàng).")
    add_paragraph_styled(doc, "• WarehouseStaff 3 - ImportSlip 3: 1 - n; Supplier 3 - ImportSlip 3: 1 - n.")
    add_paragraph_styled(doc, "• ImportSlip 3 - ImportSlipDetail 3: 1 - n (composition); Item 3 - ImportSlipDetail 3: 1 - n.")
    add_paragraph_styled(doc, "• Customer 3 - Order 3: 1 - n (Một khách hàng có nhiều đơn đặt hàng).")
    add_paragraph_styled(doc, "• Order 3 - OrderItem 3: 1 - n (composition); Item 3 - OrderItem 3: 1 - n.")
    add_paragraph_styled(doc, "• WarehouseStaff 3 - Order 3: 1 - n (Thủ kho duyệt và xuất nhiều đơn hàng).")
    add_paragraph_styled(doc, "• DeliveryStaff 3 - Order 3: 1 - n (NV giao hàng vận chuyển nhiều đơn hàng).")
    add_paragraph_styled(doc, "• Order 3 - Invoice 3: 1 - 1 (Một đơn hàng xuất kho sinh ra 1 hóa đơn thanh toán).")
    add_paragraph_styled(doc, "• WarehouseStaff 3 - Invoice 3: 1 - n (Thủ kho in nhiều hóa đơn).")
    add_paragraph_styled(doc, "• ManagementStaff 3 - Statistic 3: 1 - n (Quản lý xem nhiều báo cáo thống kê).")

    add_heading_with_spacing(doc, "1.3. Biểu đồ Use Case tổng quan & Mô tả Use Case", level=2)
    add_paragraph_styled(doc, "Biểu đồ Use Case tổng quan thể hiện toàn bộ các tác nhân và chức năng hệ thống theo đúng quy chuẩn số đuôi 3:")
    add_code_box(doc, PUML_UC_OVERALL, "Biểu đồ Use Case tổng quan hệ thống")

    add_heading_with_spacing(doc, "Bảng mô tả chi tiết các Use Case tổng quan", level=3)
    uc_headers = ["Tên Use Case", "Actor chính", "Actor phụ", "Mô tả chức năng tóm tắt"]
    add_table_styled(doc, uc_headers, OVERALL_UC_DESC, col_widths=[1.5, 1.4, 1.3, 2.3])

    add_heading_with_spacing(doc, "1.4. Biểu đồ Use Case chi tiết & Mô tả Use Case cho 2 Module", level=2)
    
    add_heading_with_spacing(doc, "1. Module 1: Customer searches for items (SearchItem 3)", level=3)
    add_paragraph_styled(doc, "Phân rã chi tiết use case tìm kiếm mặt hàng của khách hàng:")
    add_code_box(doc, PUML_UC_MOD1, "Biểu đồ Use Case chi tiết Module 1")
    mod1_headers = ["Tên Use Case con", "Actor", "Mô tả chức năng chi tiết"]
    add_table_styled(doc, mod1_headers, MOD1_UC_DESC, col_widths=[1.8, 1.4, 3.3])

    add_heading_with_spacing(doc, "2. Module 2: Warehouse staff approves orders + export (BrowseAndExportOrder 3)", level=3)
    add_paragraph_styled(doc, "Phân rã chi tiết use case duyệt và xuất đơn hàng online của thủ kho:")
    add_code_box(doc, PUML_UC_MOD2, "Biểu đồ Use Case chi tiết Module 2")
    mod2_headers = ["Tên Use Case con", "Actor", "Mô tả chức năng chi tiết"]
    add_table_styled(doc, mod2_headers, MOD2_UC_DESC, col_widths=[2.0, 1.4, 3.1])

    doc.add_page_break()

    # ------------------ ASSIGNMENT 2 ------------------
    add_heading_with_spacing(doc, "ASSIGNMENT 2: KỊCH BẢN USE CASE & MÔ HÌNH HÓA LỚP THỰC THỂ HỆ THỐNG", level=1)
    
    add_heading_with_spacing(doc, "2.1. Viết Scenario (Kịch bản) cho 2 Module theo chuẩn giáo trình", level=2)
    add_paragraph_styled(doc, "Áp dụng định dạng kịch bản chi tiết tại Mục 3.2.1 của giáo trình, bao gồm luồng chính từng bước kèm bảng dữ liệu mô phỏng và các luồng ngoại lệ:")

    add_heading_with_spacing(doc, "1. Kịch bản Module 1: Khách hàng tìm kiếm mặt hàng (SearchItem 3)", level=3)
    scen1_data = [
        ("Use Case", "SearchItem 3 (Khách hàng tìm kiếm mặt hàng)"),
        ("Actor", "Khách hàng (Customer 3)"),
        ("Tiền điều kiện", "Hệ thống hoạt động bình thường, khách hàng đã mở website siêu thị (không bắt buộc đăng nhập)."),
        ("Hậu điều kiện", "Khách hàng xem được thông tin chi tiết đầy đủ của mặt hàng mong muốn."),
        ("Kịch bản chính", 
         "1. Từ giao diện trang chủ siêu thị (GDCustomerHome 3), khách hàng bấm chọn vào thanh/menu tìm kiếm.\n"
         "2. Hệ thống hiển thị giao diện tìm kiếm mặt hàng (GDSearchItem 3) với ô nhập từ khóa tìm kiếm và các bộ lọc danh mục.\n"
         "3. Khách hàng nhập từ khóa tên sản phẩm (ví dụ: \"Sữa tươi\") vào ô tìm kiếm và bấm nút \"Tìm kiếm\".\n"
         "4. Hệ thống tiếp nhận từ khóa, thực hiện truy vấn cơ sở dữ liệu và hiển thị danh sách các mặt hàng có tên chứa từ khóa:\n"
         "   - MH001 | Sữa tươi tiệt trùng Vinamilk 1L | Hộp | 36.000 đ | Tồn kho: 150 | [Xem chi tiết]\n"
         "   - MH002 | Sữa tươi ít đường TH True Milk 1L | Hộp | 38.000 đ | Tồn kho: 85 | [Xem chi tiết]\n"
         "   - MH005 | Sữa chua uống men sống Vinamilk | Lốc | 28.000 đ | Tồn kho: 200 | [Xem chi tiết]\n"
         "5. Khách hàng click chọn vào mặt hàng \"Sữa tươi tiệt trùng Vinamilk 1L\" (MH001) để xem thông tin cụ thể.\n"
         "6. Hệ thống chuyển sang giao diện chi tiết mặt hàng (GDItemDetail 3) và hiển thị đầy đủ thông số:\n"
         "   - Mã sản phẩm: MH001\n"
         "   - Tên sản phẩm: Sữa tươi tiệt trùng nguyên chất Vinamilk 1L\n"
         "   - Danh mục: Sữa & Sản phẩm từ sữa | NSX: Vinamilk\n"
         "   - Đơn vị tính: Hộp (1000ml) | Đơn giá: 36.000 VNĐ | Tồn kho: 150 hộp\n"
         "   - Hạn sử dụng: 15/03/2027 | Mô tả: Sữa tươi 100% tiệt trùng, giàu canxi, vitamin D3.\n"
         "7. Khách hàng xem xong thông tin và có thể chọn [Quay lại danh sách] hoặc [Thêm vào giỏ hàng]."
        ),
        ("Ngoại lệ",
         "- Bước 4a: Không tìm thấy mặt hàng nào phù hợp với từ khóa: Hệ thống thông báo \"Không tìm thấy sản phẩm nào khớp với từ khóa đã nhập. Vui lòng thử từ khóa khác!\" và giữ nguyên ô tìm kiếm.\n"
         "- Bước 4b: Khách hàng để trống ô từ khóa và bấm tìm kiếm: Hệ thống cảnh báo \"Vui lòng nhập từ khóa tìm kiếm!\"."
        )
    ]
    add_table_styled(doc, ["Mục", "Nội dung kịch bản chi tiết"], scen1_data, col_widths=[1.5, 5.0])

    add_heading_with_spacing(doc, "2. Kịch bản Module 2: Thủ kho duyệt và xuất đơn hàng (BrowseAndExportOrder 3)", level=3)
    scen2_data = [
        ("Use Case", "BrowseAndExportOrder 3 (Thủ kho duyệt đơn hàng và xuất hàng cho NV giao hàng)"),
        ("Actor chính", "Thủ kho (WarehouseStaff 3)"),
        ("Actor phối hợp", "Nhân viên giao hàng (DeliveryStaff 3)"),
        ("Tiền điều kiện", "Thủ kho đã đăng nhập thành công vào hệ thống quản lý kho; trong hệ thống có đơn đặt hàng online đang ở trạng thái chưa xuất kho (\"Chờ duyệt\" hoặc \"Đã đặt\")."),
        ("Hậu điều kiện", "Đơn hàng được cập nhật trạng thái \"Đã xuất kho\", hóa đơn bán hàng được in ra, kiện hàng và hóa đơn được bàn giao thành công cho NV giao hàng."),
        ("Kịch bản chính",
         "1. Sau khi đăng nhập, từ màn hình chính của thủ kho (GDWarehouseHome 3), thủ kho bấm chọn menu \"Duyệt & Xuất đơn hàng\".\n"
         "2. Hệ thống mở giao diện duyệt đơn hàng (GDBrowseOrder 3), tự động truy vấn và hiển thị danh sách các đơn hàng online chưa xuất kho:\n"
         "   - DH101 | 28/09/2026 08:30 | Khách: Lê Văn Nam | SĐT: 0912345678 | Tổng tiền: 350.000 đ | Chờ duyệt | [Chọn duyệt]\n"
         "   - DH102 | 28/09/2026 09:15 | Khách: Trần Thị Mai | SĐT: 0987654321 | Tổng tiền: 520.000 đ | Chờ duyệt | [Chọn duyệt]\n"
         "   - DH105 | 28/09/2026 10:00 | Khách: Phạm Hoàng Long | SĐT: 0903112233 | Tổng tiền: 180.000 đ | Chờ duyệt | [Chọn duyệt]\n"
         "3. Thủ kho click chọn đơn hàng DH101 từ danh sách để xử lý.\n"
         "4. Hệ thống chuyển sang giao diện xuất kho đơn hàng (GDExportOrder 3), hiển thị chi tiết các sản phẩm cần gom trong đơn DH101, vị trí kho và tải danh sách các nhân viên giao hàng đang sẵn sàng làm việc (active):\n"
         "   - Chi tiết: 05 Hộp Sữa tươi Vinamilk 1L (Kệ A1-02); 02 Chai Dầu ăn Neptune 1L (Kệ B2-05).\n"
         "   - DS NV giao hàng sẵn sàng: [ ] NVGH01 - Vũ Tuấn Anh (Cầu Giấy); [x] NVGH03 - Nguyễn Minh Hoàng (Hà Đông).\n"
         "5. Thủ kho gom đủ hàng theo danh sách, tích chọn nhân viên giao hàng NVGH03 (Nguyễn Minh Hoàng) phụ trách vận chuyển.\n"
         "6. Thủ kho bấm nút \"Cập nhật trạng thái & Xuất kho\".\n"
         "7. Hệ thống cập nhật trạng thái đơn hàng DH101 thành \"Đã xuất kho\" (kèm mã NV giao hàng NVGH03 và thời điểm xuất), đồng thời tạo bản ghi hóa đơn thanh toán HD101 và chuyển sang giao diện in hóa đơn (GDInvoice 3).\n"
         "8. Hệ thống gửi lệnh và máy in tiến hành in hóa đơn bán hàng HD101 đầy đủ thông tin (Mã HĐ, ngày in, thông tin khách, chi tiết tiền, người lập hóa đơn, người giao hàng).\n"
         "9. Thủ kho đóng gói kiện hàng và bàn giao trực tiếp kiện hàng cùng hóa đơn giấy đã in cho NV giao hàng Nguyễn Minh Hoàng mang đi giao. Hệ thống thông báo hoàn tất và quay lại danh sách đơn hàng."
        ),
        ("Ngoại lệ",
         "- Bước 2a: Không có đơn hàng nào ở trạng thái chưa xuất: Hệ thống thông báo \"Hiện tại không có đơn hàng nào cần xuất kho!\".\n"
         "- Bước 4a: Hàng trong kho thực tế bị thiếu/hỏng: Thủ kho bấm \"Báo thiếu hàng/Liên hệ khách\", đơn hàng được chuyển sang trạng thái chờ xử lý tồn kho.\n"
         "- Bước 5a: Không có nhân viên giao hàng nào đang online/sẵn sàng: Hệ thống báo \"Chưa có nhân viên giao hàng khả dụng. Vui lòng phân công sau!\"."
        )
    ]
    add_table_styled(doc, ["Mục", "Nội dung kịch bản chi tiết"], scen2_data, col_widths=[1.5, 5.0])

    add_heading_with_spacing(doc, "2.2. Dựng sơ đồ lớp thực thể của HỆ THỐNG theo 5 bước phương pháp trích danh từ", level=2)
    add_paragraph_styled(doc, "Thực hiện phương pháp trích danh từ chuẩn xác theo 5 bước tại Mục 3.2.2 của giáo trình:")

    add_heading_with_spacing(doc, "Bước 1: Mô tả ngắn gọn nhưng đầy đủ hệ thống trong một đoạn văn", level=3)
    add_paragraph_styled(doc, "\"Hệ thống là một trang web hỗ trợ quản lý siêu thị trực tuyến và bán lẻ đa kênh. Trong đó, khách hàng có thể đăng ký tài khoản thành viên, tra cứu tìm kiếm các mặt hàng theo từ khóa tên, xem thông tin chi tiết từng mặt hàng về đơn giá, số lượng tồn kho, danh mục, nhà cung cấp, hoặc đặt các đơn đặt hàng trực tuyến giao tận nơi, cũng như mua hàng trực tiếp tại quầy thanh toán. Nhân viên bán hàng phụ trách bán hàng tại quầy và in hóa đơn thanh toán cho khách hàng mua trực tiếp. Nhân viên thủ kho có nhiệm vụ nhập hàng hóa từ các nhà cung cấp theo các phiếu nhập hàng ghi rõ chi tiết số lượng và giá nhập; quản lý thông tin các mặt hàng và danh bạ đối tác cung cấp; kiểm tra danh sách đơn đặt hàng online chưa xuất kho, lựa chọn nhân viên giao hàng phụ trách, cập nhật trạng thái xuất kho cho đơn hàng, in hóa đơn bán hàng hợp lệ và bàn giao kiện hàng cùng hóa đơn cho nhân viên giao hàng mang đi phát cho khách. Nhân viên quản lý có quyền thiết lập các tham số hệ thống chung và theo dõi các báo cáo thống kê chuyên sâu gồm: thống kê mặt hàng bán chạy và tồn kho, thống kê tình hình nhập hàng theo từng nhà cung cấp, và thống kê doanh thu lợi nhuận theo các mốc thời gian.\"", italic=True)

    add_heading_with_spacing(doc, "Bước 2: Trích các danh từ xuất hiện trong đoạn văn", level=3)
    add_paragraph_styled(doc, "• Danh từ người: Khách hàng, thành viên, tài khoản, nhân viên bán hàng, nhân viên thủ kho, nhà cung cấp, nhân viên giao hàng, nhân viên quản lý, người dùng, người nhận.")
    add_paragraph_styled(doc, "• Danh từ cơ sở, vật chất: Siêu thị, trang web, kho hàng, quầy thanh toán, kệ hàng, kiện hàng, xe giao hàng, địa chỉ, số điện thoại, máy in.")
    add_paragraph_styled(doc, "• Danh từ nghiệp vụ, thông tin: Mặt hàng, hàng hóa, từ khóa, tên mặt hàng, đơn giá, số lượng tồn kho, danh mục, đơn đặt hàng, chi tiết đơn hàng, trạng thái đơn hàng, phiếu nhập hàng, chi tiết đơn nhập, giá nhập, hóa đơn bán hàng, hóa đơn thanh toán, tiền thừa, chiết khấu, báo cáo thống kê, thống kê mặt hàng, thống kê nhà cung cấp, thống kê doanh thu, thời gian, doanh thu, lợi nhuận.")

    add_heading_with_spacing(doc, "Bước 3: Đánh giá và lựa chọn danh từ làm lớp thực thể hoặc thuộc tính", level=3)
    add_paragraph_styled(doc, "• Loại bỏ danh từ trừu tượng: Hệ thống, trang web, phần mềm, từ khóa, kiện hàng, máy in, tiền thừa, chiết khấu, thời gian -> Loại.")
    add_paragraph_styled(doc, "• Đề xuất Lớp thực thể (Tên tiếng Anh + số đuôi 3):")
    add_paragraph_styled(doc, "  - User 3 (trừu tượng): username, password, email, phone, role, note.")
    add_paragraph_styled(doc, "  - FullName 3: firstName, middleName, lastName; Address 3: houseNumber, street, ward, district, city.")
    add_paragraph_styled(doc, "  - Customer 3 (kế thừa User 3): customerCode, membershipLevel, rewardPoints.")
    add_paragraph_styled(doc, "  - Staff 3 (trừu tượng, kế thừa User 3): staffCode, position, salaryRate.")
    add_paragraph_styled(doc, "  - WarehouseStaff 3 (kế thừa Staff 3): warehouseArea.")
    add_paragraph_styled(doc, "  - Saler 3 (kế thừa Staff 3): counterNumber.")
    add_paragraph_styled(doc, "  - ManagementStaff 3 (kế thừa Staff 3): department.")
    add_paragraph_styled(doc, "  - DeliveryStaff 3 (kế thừa Staff 3): vehicleNumber, deliveryArea, deliveryStatus.")
    add_paragraph_styled(doc, "  - Supplier 3: code, name, taxCode, phone, email, bankAccount, note.")
    add_paragraph_styled(doc, "  - Category 3: code, name, description.")
    add_paragraph_styled(doc, "  - Item 3: code, name, unit, price, description, stockQuantity, origin, expiryDate.")
    add_paragraph_styled(doc, "  - ImportSlip 3: code, importDate, totalAmount, note.")
    add_paragraph_styled(doc, "  - ImportSlipDetail 3: quantity, importPrice, amount.")
    add_paragraph_styled(doc, "  - Order 3: code, orderDate, status, totalAmount, deliveryAddress, recipientPhone, note.")
    add_paragraph_styled(doc, "  - OrderItem 3: quantity, unitPrice, discount, amount.")
    add_paragraph_styled(doc, "  - Invoice 3: code, createdDate, paymentMethod, totalAmount, paymentStatus, note.")
    add_paragraph_styled(doc, "  - Statistic 3 (trừu tượng): startDate, endDate, description.")
    add_paragraph_styled(doc, "  - ItemStatistic 3 (kế thừa Statistic 3): totalQuantitySold, totalRevenue.")
    add_paragraph_styled(doc, "  - SupplierStatistic 3 (kế thừa Statistic 3): totalImportAmount, totalImportSlips.")
    add_paragraph_styled(doc, "  - RevenueStatistic 3 (kế thừa Statistic 3): totalOrders, totalRevenue, netProfit.")

    add_heading_with_spacing(doc, "Bước 4: Xác định quan hệ số lượng giữa các thực thể", level=3)
    add_paragraph_styled(doc, "• User 3 - FullName 3: 1 - 1; User 3 - Address 3: 1 - 1; Supplier 3 - Address 3: 1 - 1.")
    add_paragraph_styled(doc, "• Category 3 - Item 3: 1 - n.")
    add_paragraph_styled(doc, "• WarehouseStaff 3 - ImportSlip 3: 1 - n; Supplier 3 - ImportSlip 3: 1 - n.")
    add_paragraph_styled(doc, "• ImportSlip 3 - Item 3 là quan hệ nhiều - nhiều (n - n): Tách thành 2 quan hệ 1 - n qua lớp trung gian ImportSlipDetail 3.")
    add_paragraph_styled(doc, "• Customer 3 - Order 3: 1 - n.")
    add_paragraph_styled(doc, "• Order 3 - Item 3 là quan hệ nhiều - nhiều (n - n): Tách thành 2 quan hệ 1 - n qua lớp trung gian OrderItem 3.")
    add_paragraph_styled(doc, "• WarehouseStaff 3 - Order 3: 1 - n; DeliveryStaff 3 - Order 3: 1 - n.")
    add_paragraph_styled(doc, "• Order 3 - Invoice 3: 1 - 1; WarehouseStaff 3 - Invoice 3: 1 - n.")
    add_paragraph_styled(doc, "• ManagementStaff 3 - Statistic 3: 1 - n.")

    add_heading_with_spacing(doc, "Bước 5: Xác định quan hệ đối tượng giữa các thực thể", level=3)
    add_paragraph_styled(doc, "• Kế thừa (Generalization): Customer 3, Staff 3 kế thừa User 3; WarehouseStaff 3, Saler 3, ManagementStaff 3, DeliveryStaff 3 kế thừa Staff 3; ItemStatistic 3, SupplierStatistic 3, RevenueStatistic 3 kế thừa Statistic 3.")
    add_paragraph_styled(doc, "• Gắn chặt / Hợp thành (Composition): User 3 sở hữu FullName 3 và Address 3; ImportSlip 3 chứa ImportSlipDetail 3; Order 3 chứa OrderItem 3.")
    add_paragraph_styled(doc, "• Kết tập (Aggregation): Category 3 tập hợp Item 3.")
    add_paragraph_styled(doc, "• Liên kết (Association): WarehouseStaff 3 - ImportSlip 3, Supplier 3 - ImportSlip 3, WarehouseStaff 3 - Order 3, DeliveryStaff 3 - Order 3, Order 3 - Invoice 3.")

    add_heading_with_spacing(doc, "Biểu đồ lớp thực thể pha phân tích toàn hệ thống (PlantUML)", level=3)
    add_code_box(doc, PUML_ENTITY_SYSTEM, "Biểu đồ lớp thực thể phân tích hệ thống")

    doc.add_page_break()

    # ------------------ ASSIGNMENT 3 ------------------
    add_heading_with_spacing(doc, "ASSIGNMENT 3: THIẾT KẾ PHÂN TÍCH TĨNH & HOẠT ĐỘNG CHI TIẾT MODUL", level=1)
    
    add_heading_with_spacing(doc, "3.1. Trích và vẽ biểu đồ lớp cho 2 Module (Static Analysis Class Diagram)", level=2)
    add_paragraph_styled(doc, "Áp dụng chặt chẽ quy trình trích lớp biên và gán phương thức nghiệp vụ theo Mục 3.2.3 của giáo trình:")
    add_paragraph_styled(doc, "• Bước 1: Mỗi giao diện tương tác đề xuất thành 1 lớp biên (tiền tố GD).")
    add_paragraph_styled(doc, "• Bước 2: Đặt tên thuộc tính giao diện với tiền tố chuẩn: in... (dữ liệu vào), out... (dữ liệu ra), sub... (hành động submit).")
    add_paragraph_styled(doc, "• Bước 3: Đề xuất phương thức và gán lớp thực thể theo 2 quy tắc vàng:")
    add_paragraph_styled(doc, "  - Quy tắc 1: Nếu tham số ra (output) liên quan lớp thực thể nào thì gán cho lớp đó.")
    add_paragraph_styled(doc, "  - Quy tắc 2: Nếu tham số ra không quyết định, gán cho lớp thực thể nhỏ nhất chứa nhiều nhất các tham số vào (input).")

    add_heading_with_spacing(doc, "1. Phân tích tĩnh Module 1: Customer searches for items (SearchItem 3)", level=3)
    add_paragraph_styled(doc, "• Lớp biên đề xuất:")
    add_paragraph_styled(doc, "  - GDCustomerHome 3: subSearchItem, subOrderOnline, subLogin.")
    add_paragraph_styled(doc, "  - GDSearchItem 3: inSearchKeyword, outItemList, subSearch, subSelectItem, subBack.")
    add_paragraph_styled(doc, "  - GDItemDetail 3: outItemDetail, subAddToCart, subBack.")
    add_paragraph_styled(doc, "• Đề xuất phương thức nghiệp vụ:")
    add_paragraph_styled(doc, "  - Tìm kiếm mặt hàng theo từ khóa: input là keyword (String), output là List<Item 3> -> Gán searchItemByName(keyword) cho lớp Item 3.")
    add_paragraph_styled(doc, "  - Lấy chi tiết mặt hàng: input là itemId (String), output là Item 3 -> Gán getItemDetail(itemId) cho lớp Item 3.")
    add_code_box(doc, PUML_CLASS_MOD1, "Biểu đồ lớp phân tích Module 1")

    add_heading_with_spacing(doc, "2. Phân tích tĩnh Module 2: Warehouse staff approves orders + export (BrowseAndExportOrder 3)", level=3)
    add_paragraph_styled(doc, "• Lớp biên đề xuất:")
    add_paragraph_styled(doc, "  - GDWarehouseHome 3: subBrowseOrder, subImportGoods, subManageItem, subManageSupplier.")
    add_paragraph_styled(doc, "  - GDBrowseOrder 3: outUnexportedOrders, subSelectOrder, subRefresh.")
    add_paragraph_styled(doc, "  - GDExportOrder 3: outOrderDetail, outDeliveryStaffList, inSelectedDeliveryStaff, subUpdateExport, subBack.")
    add_paragraph_styled(doc, "  - GDInvoice 3: outInvoiceData, subPrint, subFinish.")
    add_paragraph_styled(doc, "• Đề xuất phương thức nghiệp vụ:")
    add_paragraph_styled(doc, "  - Lấy danh sách đơn hàng online chưa xuất kho: input: None, output: List<Order 3> -> Gán getUnexportedOrders() cho lớp Order 3.")
    add_paragraph_styled(doc, "  - Lấy chi tiết một đơn đặt hàng: input: orderId (String), output: Order 3 -> Gán getOrderDetail(orderId) cho lớp Order 3.")
    add_paragraph_styled(doc, "  - Lấy danh sách nhân viên giao hàng sẵn sàng: input: None, output: List<DeliveryStaff 3> -> Gán getAvailableDeliveryStaff() cho lớp DeliveryStaff 3.")
    add_paragraph_styled(doc, "  - Cập nhật trạng thái đơn hàng đã xuất kho: input: orderId, deliveryStaffId, output: boolean -> Gán updateExportStatus(orderId, deliveryStaffId) cho lớp Order 3.")
    add_paragraph_styled(doc, "  - Tạo và lấy dữ liệu hóa đơn bán lẻ: input: orderId, warehouseStaffId, output: Invoice 3 -> Gán createInvoice(orderId, warehouseStaffId) và getInvoiceData(invoiceId) cho lớp Invoice 3.")
    add_code_box(doc, PUML_CLASS_MOD2, "Biểu đồ lớp phân tích Module 2")

    add_heading_with_spacing(doc, "3.2. Biểu đồ chuyển trạng thái (State Diagram) cho 2 Module", level=2)
    add_paragraph_styled(doc, "Theo Mục 3.2.4 của giáo trình: Mỗi trạng thái chờ (Wait State) tương ứng với một lần hệ thống hiển thị một giao diện (lớp biên) để chờ người dùng tương tác; điều kiện chuyển trạng thái là hành động của người dùng trên giao diện đó.")

    add_heading_with_spacing(doc, "1. Biểu đồ trạng thái Module 1: Customer searches for items", level=3)
    add_code_box(doc, PUML_STATE_MOD1, "Biểu đồ chuyển trạng thái Module 1")

    add_heading_with_spacing(doc, "2. Biểu đồ trạng thái Module 2: Warehouse staff approves orders + export", level=3)
    add_code_box(doc, PUML_STATE_MOD2, "Biểu đồ chuyển trạng thái Module 2")

    add_heading_with_spacing(doc, "3.3. Viết kịch bản chi tiết (ver 2.0) cho 2 Module", level=2)
    add_paragraph_styled(doc, "Kịch bản phiên bản 2 (ver 2.0) mô tả tuần tự các bước tương tác giữa Tác nhân, Lớp biên và Lớp thực thể:")

    add_heading_with_spacing(doc, "1. Kịch bản chi tiết ver 2.0 cho Module 1: Customer searches for items", level=3)
    add_paragraph_styled(doc, "1. Khách hàng (Customer 3) bấm chọn vào menu tìm kiếm trên màn hình chính (GDCustomerHome 3).")
    add_paragraph_styled(doc, "2. Lớp GDCustomerHome 3 khởi tạo và gọi hiển thị lớp biên GDSearchItem 3.")
    add_paragraph_styled(doc, "3. Lớp GDSearchItem 3 hiển thị ô tìm kiếm cho khách hàng.")
    add_paragraph_styled(doc, "4. Khách hàng nhập từ khóa tên sản phẩm và bấm nút Tìm kiếm (enterKeywordAndSubmit).")
    add_paragraph_styled(doc, "5. Lớp GDSearchItem 3 gửi thông điệp yêu cầu tìm kiếm tới lớp thực thể Item 3 thông qua phương thức searchItemByName(keyword).")
    add_paragraph_styled(doc, "6. Lớp thực thể Item 3 truy vấn cơ sở dữ liệu và tìm các mặt hàng có tên chứa từ khóa.")
    add_paragraph_styled(doc, "7. Lớp Item 3 trả về danh sách các đối tượng mặt hàng (itemList) cho lớp biên GDSearchItem 3.")
    add_paragraph_styled(doc, "8. Lớp GDSearchItem 3 định dạng và hiển thị bảng kết quả tìm kiếm cho khách hàng.")
    add_paragraph_styled(doc, "9. Khách hàng click chọn một mặt hàng trong danh sách để xem chi tiết (selectItem).")
    add_paragraph_styled(doc, "10. Lớp GDSearchItem 3 khởi tạo và gọi hiển thị lớp biên GDItemDetail 3.")
    add_paragraph_styled(doc, "11. Lớp GDItemDetail 3 gọi lớp thực thể Item 3 yêu cầu thông tin chi tiết qua phương thức getItemDetail(itemId).")
    add_paragraph_styled(doc, "12. Lớp Item 3 lấy thông tin chi tiết đầy đủ của mặt hàng và trả kết quả về cho GDItemDetail 3.")
    add_paragraph_styled(doc, "13. Lớp GDItemDetail 3 hiển thị toàn bộ thông tin chi tiết của mặt hàng cho khách hàng xem.")
    add_paragraph_styled(doc, "14. Khách hàng xem xong thông tin và có thể bấm nút quay lại hoặc thêm sản phẩm vào giỏ.")

    add_heading_with_spacing(doc, "2. Kịch bản chi tiết ver 2.0 cho Module 2: Warehouse staff approves orders + export", level=3)
    add_paragraph_styled(doc, "1. Thủ kho (WarehouseStaff 3) sau khi đăng nhập, tại màn hình chính GDWarehouseHome 3, click chọn menu duyệt đơn hàng (clickBrowseOrders).")
    add_paragraph_styled(doc, "2. Lớp GDWarehouseHome 3 gọi khởi tạo và hiển thị lớp biên GDBrowseOrder 3.")
    add_paragraph_styled(doc, "3. Lớp GDBrowseOrder 3 gửi thông điệp tới lớp thực thể Order 3 gọi phương thức getUnexportedOrders() để lấy danh sách đơn chưa xuất.")
    add_paragraph_styled(doc, "4. Lớp thực thể Order 3 truy vấn và trích xuất tất cả các đơn hàng có trạng thái chưa xuất kho.")
    add_paragraph_styled(doc, "5. Lớp Order 3 trả kết quả danh sách đơn hàng (unexportedList) về cho lớp GDBrowseOrder 3.")
    add_paragraph_styled(doc, "6. Lớp GDBrowseOrder 3 hiển thị danh sách các đơn hàng chưa xuất lên bảng cho thủ kho quan sát.")
    add_paragraph_styled(doc, "7. Thủ kho click chọn một đơn hàng cụ thể từ danh sách (selectUnexportedOrder).")
    add_paragraph_styled(doc, "8. Lớp GDBrowseOrder 3 gọi khởi tạo và hiển thị giao diện xuất kho GDExportOrder 3.")
    add_paragraph_styled(doc, "9. Lớp GDExportOrder 3 gọi phương thức getOrderDetail(orderId) trên lớp thực thể Order 3 để lấy danh sách chi tiết các mặt hàng cần gom.")
    add_paragraph_styled(doc, "10. Lớp Order 3 trả thông tin chi tiết đơn hàng về cho GDExportOrder 3.")
    add_paragraph_styled(doc, "11. Lớp GDExportOrder 3 gửi thông điệp tới lớp thực thể DeliveryStaff 3 gọi phương thức getAvailableDeliveryStaff() để lấy danh sách NV giao hàng đang rảnh.")
    add_paragraph_styled(doc, "12. Lớp DeliveryStaff 3 trả danh sách nhân viên giao hàng khả dụng về cho GDExportOrder 3.")
    add_paragraph_styled(doc, "13. Lớp GDExportOrder 3 hiển thị chi tiết đơn hàng và danh sách NV giao hàng cho thủ kho lựa chọn.")
    add_paragraph_styled(doc, "14. Thủ kho chọn nhân viên giao hàng phụ trách và bấm nút xác nhận xuất kho (selectStaffAndUpdateStatus).")
    add_paragraph_styled(doc, "15. Lớp GDExportOrder 3 gọi phương thức updateExportStatus(orderId, staffId) trên lớp Order 3.")
    add_paragraph_styled(doc, "16. Lớp Order 3 cập nhật trạng thái đơn hàng thành \"Đã xuất kho\", ghi nhận thời gian và mã NV giao hàng, trả kết quả thành công cho GDExportOrder 3.")
    add_paragraph_styled(doc, "17. Lớp GDExportOrder 3 gọi phương thức createInvoice(orderId, warehouseStaffId) trên lớp thực thể Invoice 3 để sinh hóa đơn.")
    add_paragraph_styled(doc, "18. Lớp Invoice 3 khởi tạo bản ghi hóa đơn và trả dữ liệu hóa đơn về cho GDExportOrder 3.")
    add_paragraph_styled(doc, "19. Lớp GDExportOrder 3 chuyển tiếp dữ liệu và hiển thị giao diện hóa đơn GDInvoice 3.")
    add_paragraph_styled(doc, "20. Lớp GDInvoice 3 hiển thị hóa đơn, thủ kho bấm lệnh in (printAndDeliver); hệ thống in hóa đơn ra máy in, thủ kho bàn giao hàng và hóa đơn cho nhân viên giao hàng.")

    doc.add_page_break()

    # ------------------ ASSIGNMENT 4 ------------------
    add_heading_with_spacing(doc, "ASSIGNMENT 4: THIẾT KẾ HOẠT ĐỘNG (COMMUNICATION DIAGRAM) & TỔNG HỢP TOÀN BỘ PHA PHÂN TÍCH", level=1)
    
    add_heading_with_spacing(doc, "4.1. Biểu đồ giao tiếp (Communication Diagram) cho 2 Module", level=2)
    add_paragraph_styled(doc, "Biểu đồ giao tiếp thể hiện trực quan các đối tượng (Tác nhân, Lớp biên, Lớp thực thể) và luồng trao đổi thông điệp với số thứ tự tương ứng 100% với các bước trong Kịch bản chi tiết ver 2.0:")

    add_heading_with_spacing(doc, "1. Biểu đồ giao tiếp Module 1: Customer searches for items", level=3)
    add_code_box(doc, PUML_COMM_MOD1, "Biểu đồ giao tiếp Module 1")

    add_heading_with_spacing(doc, "2. Biểu đồ giao tiếp Module 2: Warehouse staff approves orders + export", level=3)
    add_code_box(doc, PUML_COMM_MOD2, "Biểu đồ giao tiếp Module 2")

    add_heading_with_spacing(doc, "4.2. Bảng ma trận truy vết (Traceability Matrix) & Tổng kết pha phân tích", level=2)
    add_paragraph_styled(doc, "Bảng ma trận truy vết chứng minh tính toàn vẹn 100% Traceability của toàn bộ các phần tử trong pha phân tích:")

    mat_headers = ["Từ khóa chuyên môn", "Use Case", "Kịch bản (Scenario)", "Lớp thực thể", "Lớp biên & Phương thức", "Kịch bản v2.0 & Biểu đồ giao tiếp"]
    mat_data = [
        ("Khách hàng (Customer 3)", "SearchItem 3", "Scenario Mod 1 (7 bước)", "Customer 3", "GDCustomerHome 3\nGDSearchItem 3", "Thông điệp 1, 2, 3, 6, 7, 12"),
        ("Mặt hàng (Item 3)\nDanh mục (Category 3)", "SearchItem 3\nViewItemDetail 3", "Mockup DS kết quả\nMockup chi tiết MH", "Item 3\nCategory 3", "searchItemByName()\ngetItemDetail()\nGDItemDetail 3", "Thông điệp 4, 5, 8, 9, 10, 11"),
        ("Thủ kho (WarehouseStaff 3)", "BrowseAndExportOrder 3", "Scenario Mod 2 (9 bước)", "WarehouseStaff 3", "GDWarehouseHome 3\nGDBrowseOrder 3", "Thông điệp 1, 2, 6, 13, 20"),
        ("Đơn đặt hàng (Order 3)\nChi tiết đơn (OrderItem 3)", "BrowseAndExportOrder 3\nSelectUnexportedOrder 3", "Mockup DS đơn chưa xuất\nMockup chi tiết đơn", "Order 3\nOrderItem 3", "getUnexportedOrders()\ngetOrderDetail()\nupdateExportStatus()", "Thông điệp 3, 4, 5, 8, 9, 14, 15, 16"),
        ("NV giao hàng (DeliveryStaff 3)", "AssignDeliveryStaff 3\nHandoverGoodsAndInvoice 3", "Mockup chọn NV giao hàng phụ trách", "DeliveryStaff 3", "getAvailableDeliveryStaff()\nGDExportOrder 3", "Thông điệp 10, 11, 12"),
        ("Hóa đơn bán hàng (Invoice 3)", "PrintInvoice 3", "Mẫu in hóa đơn bán lẻ", "Invoice 3", "createInvoice()\ngetInvoiceData()\nGDInvoice 3", "Thông điệp 17, 18, 19, 20")
    ]
    add_table_styled(doc, mat_headers, mat_data, col_widths=[1.1, 1.1, 1.2, 0.9, 1.2, 1.0])

    add_heading_with_spacing(doc, "Kết luận pha phân tích:", level=3)
    add_paragraph_styled(doc, "1. Báo cáo đã hoàn thành trọn vẹn và nhất quán 100% cả 4 Assignment theo đúng case study chuẩn sách giáo trình.")
    add_paragraph_styled(doc, "2. Mọi quy chuẩn kỹ thuật của đề bài (Số đuôi 3 cho Use Case và Tên Class, mã đề 09, mã sinh viên D23DCVT103) đã được tuân thủ nghiêm ngặt.")
    add_paragraph_styled(doc, "3. Toàn bộ các phân đoạn biểu đồ đều được xuất bản thành mã nguồn PlantUML chuẩn xác, sẵn sàng chuyển giao sang pha Thiết kế chi tiết (Chương 4) và Cài đặt (Chương 5).")

    # Save docx
    docx_path = "D:\\school_project\\PTTK\\09D23DCVT103.docx"
    doc.save(docx_path)
    print(f"File {docx_path} saved successfully!")

# Run docx build
build_docx()
