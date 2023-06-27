///// Sketch 8

var sketch8 = function(p) {
    var mouse1 = 0, mouse2 = 0, mouse3 = 0, mouse4 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg8_1111 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-0000.png');
	bg8_1110 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-0001.png');
	bg8_1101 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-0010.png');
	bg8_1100 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-0011.png');
	bg8_1011 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-0100.png');
	bg8_1010 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-0101.png');
	bg8_1001 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-0110.png');
	bg8_1000 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-0111.png');
	bg8_0111 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-1000.png');
	bg8_0110 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-1001.png');
	bg8_0101 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-1010.png');
	bg8_0100 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-1011.png');
	bg8_0011 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-1100.png');
	bg8_0010 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-1101.png');
	bg8_0001 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-1110.png');
	bg8_0000 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier8-1111.png');
	
    }

    p.draw = function() {
	str1 = mouse1.toString();
	str2 = mouse2.toString();
	str3 = mouse3.toString();
	str4 = mouse4.toString();
	picName = "bg8_" + str1 + str2 + str3 + str4;
	p.background(eval(picName));
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 72) && (p.mouseX < 160) && (p.mouseY > 20) && (p.mouseY < 97)) {
	    mouse1 = (mouse1 + 1)%2;
	}
	if ((p.mouseX > 285) && (p.mouseX < 373) && (p.mouseY > 20) && (p.mouseY < 97)) {
	    mouse2 = (mouse2 + 1)%2;
	}
	if ((p.mouseX > 72) && (p.mouseX < 160) && (p.mouseY > 115) && (p.mouseY < 192)) {
	    mouse3 = (mouse3 + 1)%2;
	}
	if ((p.mouseX > 285) && (p.mouseX < 373) && (p.mouseY > 115) && (p.mouseY < 192)) {
	    mouse4 = (mouse4 + 1)%2;
	}
    }
}

var myp5 = new p5(sketch8, 'sketch8');


///// Sketch 9

var sketch9 = function(p) {
    var mouse1 = 0, mouse2 = 0, mouse3 = 0, mouse4 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg9_1111 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-0000.png');
	bg9_1110 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-0001.png');
	bg9_1101 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-0010.png');
	bg9_1100 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-0011.png');
	bg9_1011 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-0100.png');
	bg9_1010 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-0101.png');
	bg9_1001 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-0110.png');
	bg9_1000 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-0111.png');
	bg9_0111 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-1000.png');
	bg9_0110 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-1001.png');
	bg9_0101 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-1010.png');
	bg9_0100 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-1011.png');
	bg9_0011 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-1100.png');
	bg9_0010 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-1101.png');
	bg9_0001 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-1110.png');
	bg9_0000 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier9-1111.png');
	
    }

    p.draw = function() {
	str1 = mouse1.toString();
	str2 = mouse2.toString();
	str3 = mouse3.toString();
	str4 = mouse4.toString();
	picName = "bg9_" + str1 + str2 + str3 + str4;
	p.background(eval(picName));
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 72) && (p.mouseX < 160) && (p.mouseY > 20) && (p.mouseY < 97)) {
	    mouse1 = (mouse1 + 1)%2;
	}
	if ((p.mouseX > 285) && (p.mouseX < 373) && (p.mouseY > 20) && (p.mouseY < 97)) {
	    mouse2 = (mouse2 + 1)%2;
	}
	if ((p.mouseX > 72) && (p.mouseX < 160) && (p.mouseY > 115) && (p.mouseY < 192)) {
	    mouse3 = (mouse3 + 1)%2;
	}
	if ((p.mouseX > 285) && (p.mouseX < 373) && (p.mouseY > 115) && (p.mouseY < 192)) {
	    mouse4 = (mouse4 + 1)%2;
	}
    }
}

var myp5 = new p5(sketch9, 'sketch9');

///// Sketch 10

var sketch10 = function(p) {
    var mouse1 = 0, mouse2 = 0, mouse3 = 0, mouse4 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg10_1111 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-0000.png');
	bg10_1110 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-0001.png');
	bg10_1101 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-0010.png');
	bg10_1100 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-0011.png');
	bg10_1011 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-0100.png');
	bg10_1010 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-0101.png');
	bg10_1001 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-0110.png');
	bg10_1000 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-0111.png');
	bg10_0111 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-1000.png');
	bg10_0110 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-1001.png');
	bg10_0101 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-1010.png');
	bg10_0100 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-1011.png');
	bg10_0011 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-1100.png');
	bg10_0010 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-1101.png');
	bg10_0001 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-1110.png');
	bg10_0000 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier10-1111.png');
	
    }

    p.draw = function() {
	str1 = mouse1.toString();
	str2 = mouse2.toString();
	str3 = mouse3.toString();
	str4 = mouse4.toString();
	picName = "bg10_" + str1 + str2 + str3 + str4;
	p.background(eval(picName));
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 72) && (p.mouseX < 160) && (p.mouseY > 20) && (p.mouseY < 97)) {
	    mouse1 = (mouse1 + 1)%2;
	}
	if ((p.mouseX > 285) && (p.mouseX < 373) && (p.mouseY > 20) && (p.mouseY < 97)) {
	    mouse2 = (mouse2 + 1)%2;
	}
	if ((p.mouseX > 72) && (p.mouseX < 160) && (p.mouseY > 115) && (p.mouseY < 192)) {
	    mouse3 = (mouse3 + 1)%2;
	}
	if ((p.mouseX > 285) && (p.mouseX < 373) && (p.mouseY > 115) && (p.mouseY < 192)) {
	    mouse4 = (mouse4 + 1)%2;
	}
    }
}

var myp5 = new p5(sketch10, 'sketch10');
