package models;
import java.util.List;
import java.util.Date;

public class Item3 {
    private String code;
    private String name;
    private String unit;
    private double price;
    private String description;
    private int stockQuantity;
    private String origin;
    private Date expiryDate;
    private Category3 category;

    public List<Item3> searchItemByName(String keyword) {
        return null;
    }

    public Item3 getItemDetail(String itemId) {
        return null;
    }
}
