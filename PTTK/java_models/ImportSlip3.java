package models;
import java.util.Date;
import java.util.List;

public class ImportSlip3 {
    private String code;
    private Date importDate;
    private double totalAmount;
    private String note;
    private Supplier3 supplier;
    private WarehouseStaff3 warehouseStaff;
    private List<ImportSlipDetail3> details;
}
