package components.product.coffee;

import components.product.Coffee;
import components.product.NutritionFacts;

public class Espresso extends Coffee { // Espresso is-a Coffee, therefore the class extends to Coffee.
	public Espresso() { // Constructor.
		super(2, "Espresso", 2.45, new NutritionFacts("1 coffee cup (6fl oz)", 4, 0.21, 0.32, 0));
	}

	// Declare methods
	@Override
	public String texture() { // Overridden method.
		return "concentrated & oily";
	}

	@Override
	public String taste() { // Overridden method.
		return "a bit of bitter & also sour";
	}
}