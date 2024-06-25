package components.product.bakery;

import components.product.Bakery;

public class Bagel extends Bakery { // Bagel is-a Bakery, therefore the class extends to Bakery.
	public Bagel() { // Constructor
		super(4, "Plain Bagel", 1.95, new components.product.NutritionFacts("1 Medium Bagel (105 g)", 277, 11, 1.4, 55));
	}

	// Declare methods.
	@Override
	public String texture() { // Overridden method.
		return "a spongy & chewy";
	}

	@Override
	public String taste() { // Overridden method.
		return "a garlic-y & a little bit salty";
	}
}