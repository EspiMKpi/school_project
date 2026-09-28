package models;
import java.util.List;

public class DeliveryStaff3 extends Staff3 {
    private String vehicleNumber;
    private String deliveryArea;
    private String deliveryStatus;
    private List<Order3> deliveringOrders;

    public List<DeliveryStaff3> getAvailableDeliveryStaff() {
        return null;
    }
}
