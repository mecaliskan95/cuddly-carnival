package components.product;

import components.Product;

public abstract class Coffee extends Product { // Coffee is-a Product, therefore the class extends to Product.
	// Constructor.
	public Coffee(int id, String name, double price, NutritionFacts nutritionFacts) {
		super(id, name, price, nutritionFacts);
	}

	// Declare methods.
	@Override
	public abstract String texture(); // Abstract to force the subclass to have the same function but to be
										// overridden.

	@Override
	public String smell() { // Overridden method.
		return "strong coffee aroma";
	}

	@Override
	public abstract String taste(); // Abstract to force the subclass to have the same function but to be
									// overridden.

	@Override
	public String temperature() { // Overridden method.
		return "hot";
	}
}