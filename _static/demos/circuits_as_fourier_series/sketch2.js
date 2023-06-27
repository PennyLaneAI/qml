///// Sketch 2

var sketch2 = function(p) {
    var mouse = 0;
    var i, omega, gamma, shift;
    var gamma = 1;
    var kappa = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var steps = 100, start = 95, end = 370, lift = 232, freq = 10, amp = 30;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg2_0 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier2-0.png');
	bg2_1 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier2-1.png');
    }

    p.draw = function() {
	if (mouse == 0) {
	    p.background(bg2_0);
	    omega = -gamma;
	    shift = 0;
	}
	if (mouse == 1) {
	    p.background(bg2_1);
	    omega = gamma;
	    shift = 0;
	}
	
	let step = (end - start)/steps;
	
	for (i = 0; i < steps; i++) {
	    let x = i*step + start;
	    p.stroke(112, 206, 255);
	    p.strokeWeight(2);
	    p.line(x, lift - amp*p.cos(shift + omega*(x-start)/freq), x + step, lift - amp*p.cos(shift + omega*((x-start) + step)/freq));
	}

	for (i = 0; i < steps; i++) {
	    let x = i*step + start;
	    p.stroke(181, 242, 237);
	    p.strokeWeight(2);
	    p.line(x, lift - amp*p.sin(shift + omega*(x-start)/freq), x + step, lift - amp*p.sin(shift + omega*((x-start) + step)/freq));
	}

	if ((p.mouseX > 140) && (p.mouseX < 260) && (p.mouseY > 25) && (p.mouseY < 120)) {
	    gamma = 1 + (120 - p.mouseY)/(120 - 25);
	    kappa = 2*3.142*(260 - p.mouseX)/(260 - 140);
	}
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 140) && (p.mouseX < 260) && (p.mouseY > 25) && (p.mouseY < 120)) {
	    mouse = (mouse + 1)%2;
	}
    }
}

var myp5 = new p5(sketch2, 'sketch2');


///// Sketch 3

var sketch3 = function(p) {
    var mouse = 0;
    var i, omega, amp, shift;
    var gamma = 1;
    var kappa = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var steps = 100, start = 100, end = 370, lift = 232, freq = 10, amp0 = 30, amp1 = 30;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg3_0 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier3-0.png');
	bg3_1 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier3-1.png');
    }

    p.draw = function() {
	let step = (end - start)/steps;
	
	if (mouse == 0) {
	    p.background(bg3_0);
	    omega = -gamma;
	    shift = 0;
	    amp = amp0;
	}

	if (mouse == 1) {
	    p.background(bg3_1);
	    omega = gamma;
	    shift = 0;
	    amp = amp1;
	}

	for (i = 0; i < steps; i++) {
	    let x = i*step + start;
	    p.stroke(112, 206, 255);
	    p.strokeWeight(2);
	    p.line(x, lift - amp*p.cos(shift + omega*(x-start)/freq), x + step, lift - amp*p.cos(shift + omega*((x-start) + step)/freq));
	}

	for (i = 0; i < steps; i++) {
	    let x = i*step + start;
	    p.stroke(181, 242, 237);
	    p.strokeWeight(2);
	    p.line(x, lift - amp*p.sin(shift + omega*(x-start)/freq), x + step, lift - amp*p.sin(shift + omega*((x-start) + step)/freq));
	}

	if ((p.mouseX > 160) && (p.mouseX < 275) && (p.mouseY > 20) && (p.mouseY < 115)) {
	    gamma = 1 + (120 - p.mouseY)/(120 - 25);
	    kappa = 2*3.142*(275 - p.mouseX)/(275 - 160);
	}

	if ((p.mouseX > 135) && (p.mouseX < 160) && (p.mouseY > 20) && (p.mouseY < 115)) {
	    amp0 = -60*(p.mouseY - 115)/(115-20);
	    amp1 = 60*(p.mouseY - 20)/(115-20);
	}
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 160) && (p.mouseX < 275) && (p.mouseY > 20) && (p.mouseY < 115)) {
	    mouse = (mouse + 1)%2;
	}
    }
}

var myp5 = new p5(sketch3, 'sketch3');

///// Sketch 4

var sketch4 = function(p) {
    var mouse1 = 0, mouse2 = 0;
    var i, omega;
    var gamma = 1;
    var kappa = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var steps = 200, start = 80, end = 370, lift = 232, freq = 10, amp = 30, amp0 = 30, amp1 = 30;
    var updown = 0, shift = 0;
    
    p.setup = function() {
	const canvas1 = p.createCanvas(width, height);
	
	bg4_00 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier4-00.png');
	bg4_01 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier4-01.png');
	bg4_10 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier4-10.png');
	bg4_11 = p.loadImage('../_static/demos/circuits_as_fourier_series/src/fourier4-11.png');
    }

    p.draw = function() {
	let step = (end - start)/steps;
	
	str1 = mouse1.toString();
	str2 = mouse2.toString();
	picName = "bg4_" + str1 + str2;
	p.background(eval(picName));

	if ((p.mouseX > 107) && (p.mouseX < 187) && (p.mouseY > 40) && (p.mouseY < 100)) {
	    gamma = 1 + (100 - p.mouseY)/(100 - 40);
	    kappa = 2*3.142*(187 - p.mouseX)/(187 - 107);
	}

	if ((p.mouseX > 90) && (p.mouseX < 107) && (p.mouseY > 40) && (p.mouseY < 100)) {
	    amp0 = -15*(p.mouseY - 100)/(100-40);
	    amp1 = 15*(p.mouseY -40)/(100-40);
	    amp = amp0*amp1;
	}

	if ((p.mouseX > 210) && (p.mouseX < 235) && (p.mouseY > 45) && (p.mouseY < 90)) {
	    updown = 0.07*((p.mouseY - (45 + 90)/2) + 1.5);
	}

	if ((mouse1 + mouse2) == 0) {

	    for (i = 0; i < steps; i++) {
		let x = i*step + start;
	        p.stroke(112, 206, 255);
		p.strokeWeight(2);
		p.line(x, lift + amp0*updown, x + step, lift + amp0*updown);
	    }
	}

	if ((mouse1 + mouse2) == 2) {

	    for (i = 0; i < steps; i++) {
		let x = i*step + start;
     	        p.stroke(181, 242, 237);
		p.strokeWeight(2);
		p.line(x, lift + amp1*updown, x + step, lift + amp1*updown);
	    }
	}
	
	if ((mouse1 + mouse2) == 1) {
	    if (mouse1 == 1) {
		omega = -2*gamma;
		shift = 0;
		
	    }
	    if (mouse1 == 0) {
		omega = 2*gamma;
		shift = 0;
	    }

	    for (i = 0; i < steps; i++) {
		let x = i*step + start;
	        p.stroke(112, 206, 255);
		p.strokeWeight(2);
		p.line(x, lift - amp*p.cos(shift + omega*(x-start)/freq), x + step, lift - amp*p.cos(shift + omega*((x-start) + step)/freq));
	    }
	    
	    for (i = 0; i < steps; i++) {
		let x = i*step + start;
     	        p.stroke(181, 242, 237);
		p.strokeWeight(2);
		p.line(x, lift + amp*p.sin(shift + omega*(x-start)/freq), x + step, lift + amp*p.sin(shift + omega*((x-start) + step)/freq));
	    }
	}
    }

    p.mouseClicked = function() {
	if ((p.mouseX > 107) && (p.mouseX < 187) && (p.mouseY > 50) && (p.mouseY < 120)) {
	    mouse1 = (mouse1 + 1)%2;
	}
	if ((p.mouseX > 260) && (p.mouseX < 340) && (p.mouseY > 50) && (p.mouseY < 120)) {
	    mouse2 = (mouse2 + 1)%2;
	}
    }
}

var myp5 = new p5(sketch4, 'sketch4');
