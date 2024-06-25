package components.product.bakery;

import components.product.Bakery;

public class Croissant extends Bakery { // Croissant is-a Bakery, therefore the class extends to Bakery.
	public Croissant() { // Constructor
		super(5, "Choc. Croissant", 2.95, new components.product.NutritionFacts("1 pastry (70g)", 297, 5.7, 16, 33));
	}

	// Declare methods.
	@Override
	public String texture() { // Overridden method.
		return "buttery & flaky";
	}

	@Override
	public String taste() { // Overridden method.
		return "a delicious chocolate";
	}
}