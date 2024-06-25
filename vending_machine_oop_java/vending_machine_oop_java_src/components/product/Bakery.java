package components.product;

import components.Product;

public abstract class Bakery extends Product { // Bakery is-a Product, therefore the class extends to Product.
	// Constructor.
	public Bakery(int id, String name, double price, components.product.NutritionFacts nutritionFacts) {
		super(id, name, price, nutritionFacts);
	}

	// Declare methods.
	@Override
	public abstract String texture(); // Abstract to force the subclass to have the same function but to be
										// overridden.

	@Override
	public String smell() { // Overridden method.
		return "freshly-baked";
	}

	@Override
	public abstract String taste(); // Abstract to force the subclass to have the same function but to be
									// overridden.

	@Override
	public String temperature() { // Overridden method.
		return "warm";
	}
}