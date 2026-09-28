package models;
import java.util.List;

public class WarehouseStaff3 extends Staff3 {
    private String warehouseArea;
    private List<ImportSlip3> importSlips;
    private List<Order3> exportedOrders;
    private List<Invoice3> printedInvoices;
}
