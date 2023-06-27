///// Sketch 5

var sketch5 = function(p) {
    var mouse1 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg5_0 = p.loadImage('../_static/test_demo/src/fourier5-0.png');
	bg5_1 = p.loadImage('../_static/test_demo/src/fourier5-1.png');
    }

    p.draw = function() {
	str1 = mouse1.toString();
	picName = "bg5_" + str1;
	p.background(eval(picName));
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 120) && (p.mouseX < 240) && (p.mouseY > 40) && (p.mouseY < 145)) {
	    mouse1 = (mouse1 + 1)%2;
	}
    }
}

var myp5 = new p5(sketch5, 'sketch5');


///// Sketch 6

var sketch6 = function(p) {
    var mouse1 = 0, mouse2 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);

	bg6_00 = p.loadImage('../_static/test_demo/src/fourier6-00.png');
	bg6_01 = p.loadImage('../_static/test_demo/src/fourier6-01.png');
	bg6_10 = p.loadImage('../_static/test_demo/src/fourier6-10.png');
	bg6_11 = p.loadImage('../_static/test_demo/src/fourier6-11.png');
    }

    p.draw = function() {
	picName = "bg6_" + mouse1.toString() + mouse2.toString();
	p.background(eval(picName));
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 80) && (p.mouseX < 183) && (p.mouseY > 20) && (p.mouseY < 110)) {
	    mouse1 = (mouse1 + 1)%2;
	}
	if ((p.mouseX > 80) && (p.mouseX < 183) && (p.mouseY > 137) && (p.mouseY < 225)) {
	    mouse2 = (mouse2 + 1)%2;
	}
    }
}

var myp5 = new p5(sketch6, 'sketch6');

///// Sketch 7

var sketch7 = function(p) {
    var mouse1 = 0, mouse2 = 0, mouse3 = 0, mouse4 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg7_1111 = p.loadImage('../_static/test_demo/src/fourier7-0000.png');
	bg7_1110 = p.loadImage('../_static/test_demo/src/fourier7-0001.png');
	bg7_1101 = p.loadImage('../_static/test_demo/src/fourier7-0010.png');
	bg7_1100 = p.loadImage('../_static/test_demo/src/fourier7-0011.png');
	bg7_1011 = p.loadImage('../_static/test_demo/src/fourier7-0100.png');
	bg7_1010 = p.loadImage('../_static/test_demo/src/fourier7-0101.png');
	bg7_1001 = p.loadImage('../_static/test_demo/src/fourier7-0110.png');
	bg7_1000 = p.loadImage('../_static/test_demo/src/fourier7-0111.png');
	bg7_0111 = p.loadImage('../_static/test_demo/src/fourier7-1000.png');
	bg7_0110 = p.loadImage('../_static/test_demo/src/fourier7-1001.png');
	bg7_0101 = p.loadImage('../_static/test_demo/src/fourier7-1010.png');
	bg7_0100 = p.loadImage('../_static/test_demo/src/fourier7-1011.png');
	bg7_0011 = p.loadImage('../_static/test_demo/src/fourier7-1100.png');
	bg7_0010 = p.loadImage('../_static/test_demo/src/fourier7-1101.png');
	bg7_0001 = p.loadImage('../_static/test_demo/src/fourier7-1110.png');
	bg7_0000 = p.loadImage('../_static/test_demo/src/fourier7-1111.png');
	
    }

    p.draw = function() {
	str1 = mouse1.toString();
	str2 = mouse2.toString();
	str3 = mouse3.toString();
	str4 = mouse4.toString();
	picName = "bg7_" + str1 + str2 + str3 + str4;
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

var myp5 = new p5(sketch7, 'sketch7');
