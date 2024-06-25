package components.product.coffee;

import components.product.Coffee;
import components.product.NutritionFacts;

public class Latte extends Coffee { // Latte is-a Coffee, therefore the class extends to Coffee.
	public Latte() { // Constructor.
		super(1, "Caffe Latte", 3.35, new NutritionFacts("1 coffee cup (6fl oz)", 48, 3.3, 2.04, 4.62));
	}

	// Declare methods.
	@Override
	public String texture() { // Overridden method.
		return "fluffy";
	}

	@Override
	public String taste() { // Overridden method.
		return "mild & sweet flavor";
	}
}