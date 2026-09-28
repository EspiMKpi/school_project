package models;
import java.util.Date;
import java.util.List;

public class Order3 {
    private String code;
    private Date orderDate;
    private String status;
    private double totalAmount;
    private String deliveryAddress;
    private String recipientPhone;
    private String note;
    private Customer3 customer;
    private WarehouseStaff3 warehouseStaff;
    private DeliveryStaff3 deliveryStaff;
    private Invoice3 invoice;
    private List<OrderItem3> orderItems;

    public List<Order3> getUnexportedOrders() {
        return null;
    }

    public Order3 getOrderDetail(String orderId) {
        return null;
    }

    public boolean updateExportStatus(String orderId, String deliveryStaffId) {
        return true;
    }
}
