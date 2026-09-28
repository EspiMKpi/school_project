# -*- coding: utf-8 -*-
"""
Full Report Builder for PTTK Assignment (Subject No. 09)
Student: Nguyen Dai Dung - ID: D23DCVT103 (Digit 3)
Generates:
1. 09D23DCVT103.md
2. 09D23DCVT103.docx
"""

import sys
import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------------
# PLANTUML DEFINITIONS
# -------------------------------------------------------------

PUML_UC_OVERALL = """@startuml
left to right direction
skinparam packageStyle rectangle
skinparam actorStyle awesome
skinparam shadowing false
skinparam roundcorner 8

actor "User 3\\n(Người dùng)" as User3
actor "Member 3\\n(Thành viên)" as Member3
actor "Customer 3\\n(Khách hàng)" as Customer3
actor "Staff 3\\n(Nhân viên)" as Staff3
actor "WarehouseStaff 3\\n(Thủ kho)" as WarehouseStaff3
actor "Saler 3\\n(Nhân viên bán hàng)" as Saler3
actor "ManagementStaff 3\\n(Nhân viên quản lý)" as ManagementStaff3
actor "DeliveryStaff 3\\n(Nhân viên giao hàng)" as DeliveryStaff3

' Hierarchy of Actors
User3 <|-- Member3
User3 <|-- Customer3
Member3 <|-- Staff3
Member3 <|-- Customer3
Staff3 <|-- WarehouseStaff3
Staff3 <|-- Saler3
Staff3 <|-- ManagementStaff3
Staff3 <|-- DeliveryStaff3

rectangle "Hệ thống Quản lý Siêu thị Trực tuyến (Online Supermarket Management System)" {
  ' Common Use Cases
  usecase "Login 3" as UC_Login3
  usecase "Logout 3" as UC_Logout3
  usecase "ChangePassword 3" as UC_ChangePassword3
  
  ' Customer Use Cases
  usecase "RegisterMember 3" as UC_RegisterMember3
  usecase "SearchItem 3" as UC_SearchItem3
  usecase "OrderOnline 3" as UC_OrderOnline3
  usecase "BuyAtCounter 3" as UC_BuyAtCounter3
  
  ' Saler Use Cases
  usecase "SellAtCounter 3" as UC_SellAtCounter3
  usecase "CreateCounterInvoice 3" as UC_CreateCounterInvoice3
  
  ' Warehouse Staff Use Cases
  usecase "ImportGoods 3" as UC_ImportGoods3
  usecase "ManageItem 3" as UC_ManageItem3
  usecase "ManageSupplier 3" as UC_ManageSupplier3
  usecase "BrowseAndExportOrder 3" as UC_BrowseAndExportOrder3
  
  ' Delivery Staff Use Cases
  usecase "DeliverOrder 3" as UC_DeliverOrder3
  
  ' Management Staff Use Cases
  usecase "ManageGeneralInfo 3" as UC_ManageGeneralInfo3
  usecase "ViewStatistics 3" as UC_ViewStatistics3
  usecase "ViewItemStats 3" as UC_ViewItemStats3
  usecase "ViewSupplierStats 3" as UC_ViewSupplierStats3
  usecase "ViewRevenueStats 3" as UC_ViewRevenueStats3
}

' Actor to Use Case Connections
User3 --> UC_Login3
Member3 --> UC_Logout3
Member3 --> UC_ChangePassword3

Customer3 --> UC_RegisterMember3
Customer3 --> UC_SearchItem3
Customer3 --> UC_OrderOnline3
Customer3 --> UC_BuyAtCounter3

Saler3 --> UC_SellAtCounter3
Saler3 --> UC_CreateCounterInvoice3
UC_BuyAtCounter3 ..> UC_SellAtCounter3 : <<interact>>

WarehouseStaff3 --> UC_ImportGoods3
WarehouseStaff3 --> UC_ManageItem3
WarehouseStaff3 --> UC_ManageSupplier3
WarehouseStaff3 --> UC_BrowseAndExportOrder3
UC_BrowseAndExportOrder3 ..> DeliveryStaff3 : <<handover to>>

DeliveryStaff3 --> UC_DeliverOrder3

ManagementStaff3 --> UC_ManageGeneralInfo3
ManagementStaff3 --> UC_ViewStatistics3

UC_ViewStatistics3 <|-- UC_ViewItemStats3
UC_ViewStatistics3 <|-- UC_ViewSupplierStats3
UC_ViewStatistics3 <|-- UC_ViewRevenueStats3
@enduml"""

PUML_UC_MOD1 = """@startuml
left to right direction
skinparam packageStyle rectangle
skinparam actorStyle awesome
skinparam shadowing false
skinparam roundcorner 8

actor "Customer 3" as Customer3

rectangle "Module 1: Customer searches for items" {
  usecase "SearchItem 3\\n(Tìm kiếm mặt hàng)" as UC_SearchItem3
  usecase "SelectSearchMenu 3\\n(Chọn menu tìm kiếm)" as UC_SelectMenu3
  usecase "EnterSearchKeyword 3\\n(Nhập từ khóa tìm kiếm)" as UC_EnterKeyword3
  usecase "DisplayItemList 3\\n(Hiển thị danh sách mặt hàng)" as UC_DisplayList3
  usecase "ViewItemDetail 3\\n(Xem chi tiết mặt hàng)" as UC_ViewDetail3
  usecase "FilterByCategory 3\\n(Lọc theo danh mục)" as UC_FilterCat3
}

Customer3 --> UC_SearchItem3

UC_SearchItem3 ..> UC_SelectMenu3 : <<include>>
UC_SearchItem3 ..> UC_EnterKeyword3 : <<include>>
UC_SearchItem3 ..> UC_DisplayList3 : <<include>>
UC_ViewDetail3 ..> UC_DisplayList3 : <<extend>>
UC_FilterCat3 ..> UC_SearchItem3 : <<extend>>
@enduml"""

PUML_UC_MOD2 = """@startuml
left to right direction
skinparam packageStyle rectangle
skinparam actorStyle awesome
skinparam shadowing false
skinparam roundcorner 8

actor "WarehouseStaff 3\\n(Thủ kho)" as WarehouseStaff3
actor "DeliveryStaff 3\\n(NV Giao hàng)" as DeliveryStaff3

rectangle "Module 2: Warehouse staff approves orders + export" {
  usecase "BrowseAndExportOrder 3\\n(Duyệt và xuất đơn hàng)" as UC_BrowseExport3
  usecase "SelectBrowseOrderMenu 3\\n(Chọn menu duyệt đơn hàng)" as UC_SelectMenu3
  usecase "DisplayUnexportedOrders 3\\n(Hiển thị đơn hàng chưa xuất)" as UC_DisplayOrders3
  usecase "SelectUnexportedOrder 3\\n(Chọn đơn hàng chưa xuất)" as UC_SelectOrder3
  usecase "AssignDeliveryStaff 3\\n(Chọn NV giao hàng phụ trách)" as UC_AssignStaff3
  usecase "UpdateExportStatus 3\\n(Cập nhật trạng thái xuất đơn)" as UC_UpdateStatus3
  usecase "PrintInvoice 3\\n(In hóa đơn bán hàng)" as UC_PrintInvoice3
  usecase "HandoverGoodsAndInvoice 3\\n(Bàn giao hàng và hóa đơn)" as UC_Handover3
}

WarehouseStaff3 --> UC_BrowseExport3

UC_BrowseExport3 ..> UC_SelectMenu3 : <<include>>
UC_BrowseExport3 ..> UC_DisplayOrders3 : <<include>>
UC_BrowseExport3 ..> UC_SelectOrder3 : <<include>>
UC_BrowseExport3 ..> UC_AssignStaff3 : <<include>>
UC_BrowseExport3 ..> UC_UpdateStatus3 : <<include>>
UC_BrowseExport3 ..> UC_PrintInvoice3 : <<include>>
UC_BrowseExport3 ..> UC_Handover3 : <<include>>

UC_Handover3 ..> DeliveryStaff3 : <<interact>>
@enduml"""

PUML_ENTITY_SYSTEM = """@startuml
skinparam classAttributeIconSize 0
skinparam roundcorner 8
skinparam shadowing false

class "User 3" as User3 {
  username
  password
  role
  email
  phone
  note
}

class "FullName 3" as FullName3 {
  firstName
  middleName
  lastName
}

class "Address 3" as Address3 {
  houseNumber
  street
  ward
  district
  city
}

class "Customer 3" as Customer3 {
  customerCode
  membershipLevel
  rewardPoints
}

class "Staff 3" as Staff3 {
  staffCode
  position
  salaryRate
}

class "WarehouseStaff 3" as WarehouseStaff3 {
  warehouseArea
}

class "Saler 3" as Saler3 {
  counterNumber
}

class "ManagementStaff 3" as ManagementStaff3 {
  department
}

class "DeliveryStaff 3" as DeliveryStaff3 {
  vehicleNumber
  deliveryArea
  deliveryStatus
}

class "Supplier 3" as Supplier3 {
  code
  name
  taxCode
  phone
  email
  bankAccount
  note
}

class "Category 3" as Category3 {
  code
  name
  description
}

class "Item 3" as Item3 {
  code
  name
  unit
  price
  description
  stockQuantity
  origin
  expiryDate
}

class "ImportSlip 3" as ImportSlip3 {
  code
  importDate
  totalAmount
  note
}

class "ImportSlipDetail 3" as ImportSlipDetail3 {
  quantity
  importPrice
  amount
}

class "Order 3" as Order3 {
  code
  orderDate
  status
  totalAmount
  deliveryAddress
  recipientPhone
  note
}

class "OrderItem 3" as OrderItem3 {
  quantity
  unitPrice
  discount
  amount
}

class "Invoice 3" as Invoice3 {
  code
  createdDate
  paymentMethod
  totalAmount
  paymentStatus
  note
}

class "Statistic 3" as Statistic3 {
  startDate
  endDate
  description
}

class "ItemStatistic 3" as ItemStatistic3 {
  totalQuantitySold
  totalRevenue
}

class "SupplierStatistic 3" as SupplierStatistic3 {
  totalImportAmount
  totalImportSlips
}

class "RevenueStatistic 3" as RevenueStatistic3 {
  totalOrders
  totalRevenue
  netProfit
}

' Inheritance
User3 <|-- Customer3
User3 <|-- Staff3
Staff3 <|-- WarehouseStaff3
Staff3 <|-- Saler3
Staff3 <|-- ManagementStaff3
Staff3 <|-- DeliveryStaff3

Statistic3 <|-- ItemStatistic3
Statistic3 <|-- SupplierStatistic3
Statistic3 <|-- RevenueStatistic3

' Composition & Associations
User3 *-- "1" FullName3
User3 *-- "1" Address3
Supplier3 *-- "1" Address3

Category3 "1" o-- "0..*" Item3 : categorizes

WarehouseStaff3 "1" -- "0..*" ImportSlip3 : imports
Supplier3 "1" -- "0..*" ImportSlip3 : supplies
ImportSlip3 "1" *-- "1..*" ImportSlipDetail3 : contains
Item3 "1" -- "0..*" ImportSlipDetail3 : imported in

Customer3 "1" -- "0..*" Order3 : places
Order3 "1" *-- "1..*" OrderItem3 : contains
Item3 "1" -- "0..*" OrderItem3 : ordered in

WarehouseStaff3 "1" -- "0..*" Order3 : approves/exports
DeliveryStaff3 "1" -- "0..*" Order3 : delivers

Order3 "1" -- "1" Invoice3 : generates
WarehouseStaff3 "1" -- "0..*" Invoice3 : prints

ManagementStaff3 "1" -- "0..*" Statistic3 : views
@enduml"""

PUML_CLASS_MOD1 = """@startuml
skinparam classAttributeIconSize 0
skinparam roundcorner 8
skinparam shadowing false

class "GDCustomerHome 3" as GDCustomerHome3 <<Boundary>> {
  + subSearchItem
  + subOrderOnline
  + subLogin
}

class "GDSearchItem 3" as GDSearchItem3 <<Boundary>> {
  + inSearchKeyword
  + outItemList
  + subSearch
  + subSelectItem
  + subBack
}

class "GDItemDetail 3" as GDItemDetail3 <<Boundary>> {
  + outItemDetail
  + subAddToCart
  + subBack
}

class "Item 3" as Item3 <<Entity>> {
  - code
  - name
  - unit
  - price
  - description
  - stockQuantity
  - origin
  - expiryDate
  + searchItemByName(keyword)
  + getItemDetail(itemId)
}

class "Category 3" as Category3 <<Entity>> {
  - code
  - name
  - description
  + getAllCategories()
}

class "Customer 3" as Customer3 <<Entity>> {
  - customerCode
  - membershipLevel
}

GDCustomerHome3 ..> GDSearchItem3 : <<navigate>>
GDSearchItem3 ..> GDItemDetail3 : <<navigate>>

GDSearchItem3 ..> Item3 : <<call>> searchItemByName
GDItemDetail3 ..> Item3 : <<call>> getItemDetail

Category3 "1" o-- "0..*" Item3
Customer3 ..> GDCustomerHome3 : <<interacts>>
@enduml"""

PUML_CLASS_MOD2 = """@startuml
skinparam classAttributeIconSize 0
skinparam roundcorner 8
skinparam shadowing false

class "GDWarehouseHome 3" as GDWarehouseHome3 <<Boundary>> {
  + subBrowseOrder
  + subImportGoods
  + subManageItem
  + subManageSupplier
}

class "GDBrowseOrder 3" as GDBrowseOrder3 <<Boundary>> {
  + outUnexportedOrders
  + subSelectOrder
  + subRefresh
}

class "GDExportOrder 3" as GDExportOrder3 <<Boundary>> {
  + outOrderDetail
  + outDeliveryStaffList
  + inSelectedDeliveryStaff
  + subUpdateExport
  + subBack
}

class "GDInvoice 3" as GDInvoice3 <<Boundary>> {
  + outInvoiceData
  + subPrint
  + subFinish
}

class "Order 3" as Order3 <<Entity>> {
  - code
  - orderDate
  - status
  - totalAmount
  - deliveryAddress
  - recipientPhone
  - note
  + getUnexportedOrders()
  + getOrderDetail(orderId)
  + updateExportStatus(orderId, deliveryStaffId)
}

class "OrderItem 3" as OrderItem3 <<Entity>> {
  - quantity
  - unitPrice
  - amount
}

class "DeliveryStaff 3" as DeliveryStaff3 <<Entity>> {
  - staffCode
  - vehicleNumber
  - deliveryArea
  - deliveryStatus
  + getAvailableDeliveryStaff()
}

class "Invoice 3" as Invoice3 <<Entity>> {
  - code
  - createdDate
  - paymentMethod
  - totalAmount
  - paymentStatus
  + createInvoice(orderId, warehouseStaffId)
  + getInvoiceData(invoiceId)
}

class "WarehouseStaff 3" as WarehouseStaff3 <<Entity>> {
  - staffCode
  - warehouseArea
}

GDWarehouseHome3 ..> GDBrowseOrder3 : <<navigate>>
GDBrowseOrder3 ..> GDExportOrder3 : <<navigate>>
GDExportOrder3 ..> GDInvoice3 : <<navigate>>

GDBrowseOrder3 ..> Order3 : <<call>> getUnexportedOrders
GDExportOrder3 ..> Order3 : <<call>> getOrderDetail
GDExportOrder3 ..> DeliveryStaff3 : <<call>> getAvailableDeliveryStaff
GDExportOrder3 ..> Order3 : <<call>> updateExportStatus
GDExportOrder3 ..> Invoice3 : <<call>> createInvoice
GDInvoice3 ..> Invoice3 : <<call>> getInvoiceData

Order3 "1" *-- "1..*" OrderItem3
Order3 "1" -- "1" Invoice3
WarehouseStaff3 ..> GDWarehouseHome3 : <<interacts>>
@enduml"""

PUML_STATE_MOD1 = """@startuml
skinparam state {
  BackgroundColor White
  BorderColor Black
  ArrowColor Black
}
skinparam roundcorner 8

[*] --> WaitingCustomerHome : Khách hàng mở trang chủ

state WaitingCustomerHome : Chờ khách hàng thao tác trên màn hình chính

WaitingCustomerHome --> WaitingSearchKeyword : Khách hàng click menu tìm kiếm (clickSearchItem)

state WaitingSearchKeyword : Chờ nhập từ khóa tìm kiếm (GDSearchItem 3)

WaitingSearchKeyword --> WaitingSelectFromItemList : Khách hàng nhập từ khóa và click Tìm kiếm (enterKeywordAndSubmit)

state WaitingSelectFromItemList : Hiển thị danh sách kết quả, chờ chọn mặt hàng

WaitingSelectFromItemList --> WaitingViewItemDetail : Khách hàng click chọn một mặt hàng (selectItem)

state WaitingViewItemDetail : Hiển thị chi tiết mặt hàng (GDItemDetail 3)

WaitingViewItemDetail --> WaitingSelectFromItemList : Khách hàng click Quay lại (clickBack)
WaitingViewItemDetail --> WaitingCustomerHome : Khách hàng click Về trang chủ (clickHome)
WaitingSelectFromItemList --> WaitingCustomerHome : Khách hàng click Về trang chủ (clickHome)
WaitingCustomerHome --> [*] : Kết thúc phiên
@enduml"""

PUML_STATE_MOD2 = """@startuml
skinparam state {
  BackgroundColor White
  BorderColor Black
  ArrowColor Black
}
skinparam roundcorner 8

[*] --> WaitingWarehouseHome : Thủ kho đăng nhập, tại trang chủ kho

state WaitingWarehouseHome : Chờ chọn chức năng quản lý kho (GDWarehouseHome 3)

WaitingWarehouseHome --> WaitingSelectOrder : Thủ kho click menu duyệt đơn hàng (clickBrowseOrders)

state WaitingSelectOrder : Hiển thị danh sách đơn hàng chưa xuất, chờ chọn đơn (GDBrowseOrder 3)

WaitingSelectOrder --> WaitingSelectDeliveryStaff : Thủ kho click chọn một đơn hàng chưa xuất (selectUnexportedOrder)

state WaitingSelectDeliveryStaff : Hiển thị chi tiết đơn và DS NV giao hàng sẵn sàng, chờ gán và cập nhật (GDExportOrder 3)

WaitingSelectDeliveryStaff --> WaitingPrintInvoice : Thủ kho chọn NV giao hàng và click Xuất kho (updateExportStatus)

state WaitingPrintInvoice : Hiển thị hóa đơn bán hàng, chờ in (GDInvoice 3)

WaitingPrintInvoice --> WaitingWarehouseHome : Thủ kho click In hóa đơn & hoàn tất bàn giao (confirmPrintAndDeliver)

WaitingSelectDeliveryStaff --> WaitingSelectOrder : Thủ kho click Quay lại (clickBack)
WaitingSelectOrder --> WaitingWarehouseHome : Thủ kho click Quay lại (clickBack)
WaitingWarehouseHome --> [*] : Kết thúc phiên
@enduml"""

PUML_COMM_MOD1 = """@startuml
skinparam object {
  BackgroundColor White
  BorderColor Black
  ArrowColor Black
}
skinparam roundcorner 8
skinparam shadowing false

object "c : Customer 3" as c
object "gdHome : GDCustomerHome 3" as gdHome
object "gdSearch : GDSearchItem 3" as gdSearch
object "item : Item 3" as item
object "gdDetail : GDItemDetail 3" as gdDetail

c -> gdHome : 1: clickSearchItem()
gdHome -> gdSearch : 2: display()
c -> gdSearch : 3: enterKeywordAndSubmit(keyword)
gdSearch -> item : 4: searchItemByName(keyword)
item --> gdSearch : 5: return itemList
gdSearch --> c : 6: displayItemList(itemList)
c -> gdSearch : 7: selectItem(itemId)
gdSearch -> gdDetail : 8: display()
gdDetail -> item : 9: getItemDetail(itemId)
item --> gdDetail : 10: return itemDetail
gdDetail --> c : 11: displayItemDetail(itemDetail)
c -> gdDetail : 12: clickBackOrFinish()
@enduml"""

PUML_COMM_MOD2 = """@startuml
skinparam object {
  BackgroundColor White
  BorderColor Black
  ArrowColor Black
}
skinparam roundcorner 8
skinparam shadowing false

object "ws : WarehouseStaff 3" as ws
object "gdHome : GDWarehouseHome 3" as gdHome
object "gdBrowse : GDBrowseOrder 3" as gdBrowse
object "order : Order 3" as order
object "gdExport : GDExportOrder 3" as gdExport
object "staff : DeliveryStaff 3" as staff
object "invoice : Invoice 3" as invoice
object "gdInvoice : GDInvoice 3" as gdInvoice

ws -> gdHome : 1: clickBrowseOrders()
gdHome -> gdBrowse : 2: display()
gdBrowse -> order : 3: getUnexportedOrders()
order --> gdBrowse : 4: return unexportedList
gdBrowse --> ws : 5: displayUnexportedList(unexportedList)
ws -> gdBrowse : 6: selectUnexportedOrder(orderId)
gdBrowse -> gdExport : 7: display()
gdExport -> order : 8: getOrderDetail(orderId)
order --> gdExport : 9: return orderDetail
gdExport -> staff : 10: getAvailableDeliveryStaff()
staff --> gdExport : 11: return staffList
gdExport --> ws : 12: displayOrderDetailAndStaff(orderDetail, staffList)
ws -> gdExport : 13: selectStaffAndUpdateStatus(orderId, staffId)
gdExport -> order : 14: updateExportStatus(orderId, staffId)
order --> gdExport : 15: return success
gdExport -> invoice : 16: createInvoice(orderId, warehouseStaffId)
invoice --> gdExport : 17: return invoiceData
gdExport -> gdInvoice : 18: display()
gdInvoice --> ws : 19: displayInvoice(invoiceData)
ws -> gdInvoice : 20: printAndDeliver(invoiceId)
@enduml"""

# -------------------------------------------------------------
# 2. GLOSSARY LIST DATA
# -------------------------------------------------------------
GLOSSARY_LIST = [
    # TT, TV, EN, Giai thich
    ("1", "Thành viên", "Member 3", "Người dùng có tài khoản đã đăng ký hợp lệ và được cấp quyền đăng nhập vào hệ thống siêu thị để thực hiện các chức năng tương ứng với vai trò của mình."),
    ("2", "Người dùng", "User 3", "Khái niệm chung chỉ bất kỳ tác nhân nào tương tác với hệ thống (bao gồm cả khách vãng lai và thành viên đã có tài khoản)."),
    ("3", "Khách hàng", "Customer 3", "Người mua sắm hàng hóa tại siêu thị, có thể đăng ký tài khoản thành viên, tìm kiếm mặt hàng, đặt mua online hoặc mua trực tiếp tại quầy thu ngân."),
    ("4", "Nhân viên", "Staff 3", "Khái niệm lớp cha đại diện cho toàn bộ đội ngũ nhân sự làm việc tại siêu thị trực tuyến, có mã nhân viên, chức vụ và mức lương."),
    ("5", "Thủ kho", "WarehouseStaff 3", "Nhân viên phụ trách quản lý kho hàng: nhập hàng từ NCC, quản lý danh mục mặt hàng, NCC, duyệt đơn đặt hàng online, cập nhật trạng thái xuất kho và bàn giao hàng cho nhân viên giao hàng."),
    ("6", "Nhân viên bán hàng", "Saler 3", "Nhân viên trực tiếp bán hàng tại quầy thu ngân của siêu thị, tạo hóa đơn bán hàng trực tiếp cho khách hàng mua tại chỗ."),
    ("7", "Nhân viên quản lý", "ManagementStaff 3", "Cán bộ quản lý siêu thị, có thẩm quyền quản lý thông tin chung và xem các báo cáo thống kê chuyên sâu: thống kê mặt hàng, thống kê nhà cung cấp, thống kê doanh thu."),
    ("8", "Nhân viên giao hàng", "DeliveryStaff 3", "Nhân viên phụ trách nhận hàng hóa kèm hóa đơn từ kho sau khi duyệt để vận chuyển và giao tận nơi cho khách hàng."),
    ("9", "Nhà cung cấp", "Supplier 3", "Đơn vị hoặc đối tác cung ứng các mặt hàng, hàng hóa, nhu yếu phẩm cho siêu thị theo các hợp đồng hoặc đơn nhập hàng."),
    ("10", "Đăng nhập", "Login 3", "Hoạt động xác thực danh tính người dùng bằng tên đăng nhập và mật khẩu trước khi truy cập vào các chức năng được phân quyền."),
    ("11", "Đăng xuất", "Logout 3", "Hành động kết thúc phiên làm việc an toàn của người dùng khỏi hệ thống."),
    ("12", "Đổi mật khẩu", "ChangePassword 3", "Hoạt động cho phép thành viên thay đổi mật khẩu đăng nhập cá nhân định kỳ."),
    ("13", "Tìm kiếm mặt hàng", "SearchItem 3", "Chức năng cho phép khách hàng tra cứu danh sách các mặt hàng theo từ khóa tên mặt hàng, xem thông tin tóm tắt và chi tiết."),
    ("14", "Xem chi tiết mặt hàng", "ViewItemDetail 3", "Chức năng hiển thị toàn bộ thông số chi tiết của mặt hàng được chọn (mã, tên, đơn giá, đơn vị tính, hạn sử dụng, nhà sản xuất, tồn kho, mô tả)."),
    ("15", "Đặt hàng online", "OrderOnline 3", "Hoạt động khách hàng chọn các mặt hàng vào giỏ hàng, xác nhận địa chỉ nhận hàng, số điện thoại và gửi đơn đặt hàng trực tuyến."),
    ("16", "Duyệt đơn & Xuất kho", "BrowseAndExportOrder 3", "Quy trình thủ kho kiểm tra các đơn hàng trực tuyến chưa xuất kho, chọn nhân viên giao hàng, đổi trạng thái sang 'Đã xuất kho' và in hóa đơn."),
    ("17", "In hóa đơn", "PrintInvoice 3", "Hành động xuất bản và in chứng từ hóa đơn bán hàng hợp lệ từ đơn hàng để bàn giao kèm hàng hóa cho nhân viên giao hàng."),
    ("18", "Bán hàng tại quầy", "SellAtCounter 3", "Quy trình nhân viên bán hàng quét mã vạch sản phẩm, tính tiền và in hóa đơn trực tiếp cho khách mua tại siêu thị."),
    ("19", "Nhập hàng từ NCC", "ImportGoods 3", "Quy trình thủ kho tạo phiếu nhập hàng, ghi nhận số lượng và đơn giá nhập từ nhà cung cấp vào kho lưu trữ."),
    ("20", "Xem thống kê", "ViewStatistics 3", "Hoạt động của ban quản lý nhằm trích xuất các báo cáo tổng hợp và biểu đồ kinh doanh theo thời gian."),
    ("21", "Mặt hàng / Hàng hóa", "Item 3", "Sản phẩm được bày bán hoặc lưu trữ trong siêu thị, có mã, tên, giá bán, đơn vị tính, số lượng tồn kho, hình ảnh và danh mục."),
    ("22", "Danh mục mặt hàng", "Category 3", "Nhóm phân loại hàng hóa trong siêu thị (ví dụ: Thực phẩm tươi sống, Đồ uống, Hóa mỹ phẩm, Đồ gia dụng)."),
    ("23", "Đơn đặt hàng", "Order 3", "Chứng từ ghi nhận yêu cầu mua hàng trực tuyến của khách hàng, bao gồm ngày đặt, danh sách mặt hàng, địa chỉ giao, tổng tiền và trạng thái xử lý."),
    ("24", "Hóa đơn", "Invoice 3", "Chứng từ kế toán chính thức ghi nhận giao dịch thanh toán thành công giữa khách hàng và siêu thị."),
    ("25", "Đơn nhập hàng", "ImportSlip 3", "Phiếu nhập kho ghi nhận các mặt hàng được nhập vào siêu thị từ nhà cung cấp kèm giá nhập và số lượng thực tế.")
]

# -------------------------------------------------------------
# 3. USE CASE DESCRIPTION DATA
# -------------------------------------------------------------
OVERALL_UC_DESC = [
    ("Login 3", "User 3", "Hệ thống xác thực", "Cho phép mọi người dùng đã có tài khoản (Khách hàng, Thủ kho, NV bán hàng, Quản lý, NV giao hàng) đăng nhập vào hệ thống."),
    ("Logout 3", "Member 3", "Không có", "Cho phép thành viên kết thúc phiên làm việc hiện tại và thoát khỏi hệ thống an toàn."),
    ("ChangePassword 3", "Member 3", "Không có", "Cho phép thành viên cập nhật lại mật khẩu cá nhân để bảo vệ tài khoản."),
    ("RegisterMember 3", "Customer 3", "Không có", "Cho phép khách hàng đăng ký tài khoản thành viên mới để hưởng ưu đãi và đặt hàng trực tuyến."),
    ("SearchItem 3", "Customer 3", "Không có", "Cho phép khách hàng tìm kiếm các mặt hàng theo từ khóa tên sản phẩm và xem chi tiết."),
    ("OrderOnline 3", "Customer 3", "Không có", "Cho phép khách hàng chọn mặt hàng vào giỏ và đặt hàng trực tuyến giao tận nơi."),
    ("BuyAtCounter 3", "Customer 3", "Saler 3", "Khách hàng mua hàng trực tiếp tại quầy thanh toán của siêu thị."),
    ("SellAtCounter 3", "Saler 3", "Customer 3", "Cho phép nhân viên thu ngân thanh toán tiền và tạo hóa đơn cho khách mua tại quầy."),
    ("CreateCounterInvoice 3", "Saler 3", "Customer 3", "Tạo và in hóa đơn bán lẻ trực tiếp tại quầy."),
    ("ImportGoods 3", "WarehouseStaff 3", "Supplier 3", "Cho phép thủ kho tạo phiếu nhập hàng, ghi nhận hàng nhập từ nhà cung cấp vào kho."),
    ("ManageItem 3", "WarehouseStaff 3", "Không có", "Cho phép thủ kho quản lý thông tin các mặt hàng (thêm mới, sửa, xóa, cập nhật tồn kho)."),
    ("ManageSupplier 3", "WarehouseStaff 3", "Không có", "Cho phép thủ kho quản lý danh bạ thông tin các nhà cung cấp (thêm, sửa, xóa)."),
    ("BrowseAndExportOrder 3", "WarehouseStaff 3", "DeliveryStaff 3", "Cho phép thủ kho duyệt các đơn hàng online chưa xuất kho, chọn nhân viên giao hàng, cập nhật trạng thái đơn, in hóa đơn và bàn giao hàng."),
    ("DeliverOrder 3", "DeliveryStaff 3", "Customer 3", "Cho phép nhân viên giao hàng nhận hàng và hóa đơn từ thủ kho, vận chuyển đến khách và cập nhật kết quả giao."),
    ("ManageGeneralInfo 3", "ManagementStaff 3", "Không có", "Cho phép quản lý thiết lập các tham số hệ thống chung, thông tin siêu thị, ca làm việc."),
    ("ViewStatistics 3", "ManagementStaff 3", "Không có", "Use case tổng quát cho phép quản lý tra cứu các loại báo cáo thống kê."),
    ("ViewItemStats 3", "ManagementStaff 3", "Không có", "Cho phép quản lý xem thống kê mặt hàng bán chạy, tồn kho, tỷ lệ tiêu thụ."),
    ("ViewSupplierStats 3", "ManagementStaff 3", "Không có", "Cho phép quản lý xem thống kê nhà cung cấp theo sản lượng nhập và chi phí."),
    ("ViewRevenueStats 3", "ManagementStaff 3", "Không có", "Cho phép quản lý xem báo cáo doanh thu theo mốc thời gian (ngày, tháng, quý, năm).")
]

MOD1_UC_DESC = [
    ("SearchItem 3", "Customer 3", "Use case chính cho phép khách hàng tra cứu tìm kiếm mặt hàng theo từ khóa và xem chi tiết."),
    ("SelectSearchMenu 3", "Customer 3", "Khách hàng chọn mục 'Tìm kiếm mặt hàng' trên thanh điều hướng hoặc menu hệ thống."),
    ("EnterSearchKeyword 3", "Customer 3", "Khách hàng nhập chuỗi ký tự từ khóa đại diện cho tên mặt hàng muốn tìm vào ô tìm kiếm."),
    ("DisplayItemList 3", "Hệ thống", "Hệ thống tra cứu cơ sở dữ liệu và hiển thị danh sách các mặt hàng có tên chứa từ khóa tìm kiếm."),
    ("ViewItemDetail 3", "Customer 3", "Use case mở rộng (extend): khi khách hàng click vào một mặt hàng trong danh sách, hệ thống hiển thị chi tiết toàn diện của mặt hàng đó."),
    ("FilterByCategory 3", "Customer 3", "Use case mở rộng (extend): cho phép khách hàng thu hẹp phạm vi tìm kiếm theo phân loại danh mục hàng hóa.")
]

MOD2_UC_DESC = [
    ("BrowseAndExportOrder 3", "WarehouseStaff 3", "Use case chính cho phép thủ kho duyệt đơn hàng online, chọn NV giao hàng, cập nhật trạng thái xuất kho và in hóa đơn bàn giao."),
    ("SelectBrowseOrderMenu 3", "WarehouseStaff 3", "Thủ kho bấm chọn menu chức năng 'Duyệt & Xuất đơn hàng' trên màn hình quản lý kho."),
    ("DisplayUnexportedOrders 3", "Hệ thống", "Hệ thống truy vấn cơ sở dữ liệu và hiển thị toàn bộ danh sách các đơn hàng online đang ở trạng thái chưa xuất kho (Đã đặt / Chờ duyệt)."),
    ("SelectUnexportedOrder 3", "WarehouseStaff 3", "Thủ kho chọn một đơn hàng cụ thể từ danh sách để xem chi tiết các mặt hàng và chuẩn bị xuất kho."),
    ("AssignDeliveryStaff 3", "WarehouseStaff 3", "Thủ kho chọn nhân viên giao hàng đang sẵn sàng (active) từ danh sách để phân công giao đơn hàng này."),
    ("UpdateExportStatus 3", "WarehouseStaff 3", "Hệ thống chuyển trạng thái đơn hàng sang 'Đã xuất kho' (hoặc 'Đang giao') và lưu thời điểm xuất kho."),
    ("PrintInvoice 3", "WarehouseStaff 3", "Hệ thống tạo hóa đơn bán hàng tương ứng với đơn hàng và in hóa đơn ra máy in."),
    ("HandoverGoodsAndInvoice 3", "WarehouseStaff 3", "Thủ kho bàn giao kiện hàng thực tế cùng với hóa đơn đã in cho nhân viên giao hàng phụ trách mang đi.")
]

print("Metadata ready.")
