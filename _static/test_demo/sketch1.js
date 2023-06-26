///// Sketch 0

var sketch0_1 = function(p) {
    var mod = 0.5;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	bg0_1 = p.loadImage('../static/src/fourier0-1.png');
    }

    p.draw = function() {
	p.background(bg0_1);
    }
}

var myp5 = new p5(sketch0_1, 'sketch0_1');

var sketch0_2 = function(p) {
    var mouse = 0;
    var mod = 0.65;
    var width = 600*mod, height = 400*mod;

    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	bg0_2 = p.loadImage('../static/src/fourier0-2.png');
	bg0_3 = p.loadImage('../static/src/fourier0-3.png');
    }

    p.draw = function() {
	if (mouse == 0) {
	    p.background(bg0_2);
	}
	if (mouse == 1) {
	    p.background(bg0_3);
	}
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 0) && (p.mouseX < width) && (p.mouseY > 0) && (p.mouseY < height)) {
	    mouse = (mouse + 1)%2;
	}
    }
}

var myp5 = new p5(sketch0_2, 'sketch0_2');

///// Sketch 1

var sketch1 = function(p) {
    var mouse = 0;
    var mod = 0.65;
    var width = 600*mod, height = 400*mod;

    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	bg1_0 = p.loadImage('../static/src/fourier1-0.png');
	bg1_1 = p.loadImage('../static/src/fourier1-1.png');
    }

    p.draw = function() {
	if (mouse == 0) {
	    p.background(bg1_0);
	}
	if (mouse == 1) {
	    p.background(bg1_1);
	}
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 0) && (p.mouseX < width) && (p.mouseY > 0) && (p.mouseY < height)) {
	    mouse = (mouse + 1)%2;
	}
    }
}

var myp5 = new p5(sketch1, 'sketch1');
