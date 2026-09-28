package models;
import java.util.Date;

public class Invoice3 {
    private String code;
    private Date createdDate;
    private String paymentMethod;
    private double totalAmount;
    private String paymentStatus;
    private String note;
    private Order3 order;
    private WarehouseStaff3 warehouseStaff;

    public Invoice3 createInvoice(String orderId, String warehouseStaffId) {
        return null;
    }

    public Invoice3 getInvoiceData(String invoiceId) {
        return null;
    }
}
