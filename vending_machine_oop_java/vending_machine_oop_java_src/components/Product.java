package components;

import components.product.NutritionFacts;
import components.product.bakery.Bagel;
import components.product.bakery.Croissant;
import components.product.coffee.Cappuccino;
import components.product.coffee.Espresso;
import components.product.coffee.Latte;

public abstract class Product {

	// Declare variables.
	private String name;
	private double price;
	private int id; // ID that will shop up on the vending machine for each product.
	private NutritionFacts nutritionFacts; // Has-a relationship. Every product has-a nutrition facts.
	// Declare constructors.

	public Product(int id, String name, double price, NutritionFacts nutritionFacts) { // All-args constructor.
		this.name = name;
		this.price = price;
		this.id = id;
		this.nutritionFacts = nutritionFacts;
	}

	// Declare methods.
	public String getName() { // Getter for String name
		return name;
	}

	public double getPrice() { // Getter for double price
		return price;
	}

	public int getId() { // Getter for int id
		return id;
	}

	public NutritionFacts getNutritionFacts() { // Getter for NutritionFacts nutritionFacts
		return nutritionFacts;
	}

	public abstract String texture(); // Abstract to force the subclass to have the same function but to be
										// overridden.

	public abstract String smell(); // Abstract to force the subclass to have the same function but to be
									// overridden.

	public abstract String taste(); // Abstract to force the subclass to have the same function but to be
									// overridden.

	public abstract String temperature(); // Abstract to force the subclass to have the same function but to be
											// overridden.

	public boolean inStock(Product product, Inventory inventory) { // Checks and returns if a product is instock.
		if (product instanceof Latte) {
			if (inventory.getQtyLatte() > 0) {
				return true;
			} else {
				return false;
			}
		} else if (product instanceof Espresso) {
			if (inventory.getQtyEspresso() > 0) {
				return true;
			} else {
				return false;
			}
		} else if (product instanceof Cappuccino) {
			if (inventory.getQtyCappucino() > 0) {
				return true;
			} else {
				return false;
			}
		} else if (product instanceof Bagel) {
			if (inventory.getQtyBagel() > 0) {
				return true;
			} else {
				return false;
			}
		} else if (product instanceof Croissant) {
			if (inventory.getQtyCroissant() > 0) {
				return true;
			} else {
				return false;
			}
		} else
			return false;
	}
}