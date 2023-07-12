var url_prefix = '../_static/demos/circuits_as_fourier_series/src/';

// fetch(url_prefix + 'fourier0-1.png', {method: 'HEAD'}).then(res => {
//     if (!res.ok) {
//         url_prefix = '../_static/demos/circuits_as_fourier_series/src/';
//     }
// });


///// Sketch 0

var sketch0_1 = function(p) {
    var mod = 0.5;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    bg0_1 = p.loadImage(url_prefix + 'fourier0-1.png');
    }

    p.draw = function() {
    p.background(bg0_1);
    }
}

var myp501 = new p5(sketch0_1, 'sketch0_1');

var sketch0_2 = function(p) {
    var mouse = 0;
    var mod = 0.65;
    var width = 600*mod, height = 400*mod;

    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    bg0_2 = p.loadImage(url_prefix + 'fourier0-2.png');
    bg0_3 = p.loadImage(url_prefix + 'fourier0-3.png');
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

var myp502 = new p5(sketch0_2, 'sketch0_2');

///// Sketch 1

var sketch1 = function(p) {
    var mouse = 0;
    var mod = 0.65;
    var width = 600*mod, height = 400*mod;

    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    bg1_0 = p.loadImage(url_prefix + 'fourier1-0.png');
    bg1_1 = p.loadImage(url_prefix + 'fourier1-1.png');
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

var myp51 = new p5(sketch1, 'sketch1');


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
    
    bg2_0 = p.loadImage(url_prefix + 'fourier2-0.png');
    bg2_1 = p.loadImage(url_prefix + 'fourier2-1.png');
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

var myp52 = new p5(sketch2, 'sketch2');


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
    
    bg3_0 = p.loadImage(url_prefix + 'fourier3-0.png');
    bg3_1 = p.loadImage(url_prefix + 'fourier3-1.png');
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

var myp53 = new p5(sketch3, 'sketch3');

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
    
    bg4_00 = p.loadImage(url_prefix + 'fourier4-00.png');
    bg4_01 = p.loadImage(url_prefix + 'fourier4-01.png');
    bg4_10 = p.loadImage(url_prefix + 'fourier4-10.png');
    bg4_11 = p.loadImage(url_prefix + 'fourier4-11.png');
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

var myp54 = new p5(sketch4, 'sketch4');


///// Sketch 5

var sketch5 = function(p) {
    var mouse1 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    
    bg5_0 = p.loadImage(url_prefix + 'fourier5-0.png');
    bg5_1 = p.loadImage(url_prefix + 'fourier5-1.png');
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

var myp55 = new p5(sketch5, 'sketch5');


///// Sketch 6

var sketch6 = function(p) {
    var mouse1 = 0, mouse2 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);

    bg6_00 = p.loadImage(url_prefix + 'fourier6-00.png');
    bg6_01 = p.loadImage(url_prefix + 'fourier6-01.png');
    bg6_10 = p.loadImage(url_prefix + 'fourier6-10.png');
    bg6_11 = p.loadImage(url_prefix + 'fourier6-11.png');
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

var myp56 = new p5(sketch6, 'sketch6');

///// Sketch 7

var sketch7 = function(p) {
    var mouse1 = 0, mouse2 = 0, mouse3 = 0, mouse4 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    
    bg7_1111 = p.loadImage(url_prefix + 'fourier7-0000.png');
    bg7_1110 = p.loadImage(url_prefix + 'fourier7-0001.png');
    bg7_1101 = p.loadImage(url_prefix + 'fourier7-0010.png');
    bg7_1100 = p.loadImage(url_prefix + 'fourier7-0011.png');
    bg7_1011 = p.loadImage(url_prefix + 'fourier7-0100.png');
    bg7_1010 = p.loadImage(url_prefix + 'fourier7-0101.png');
    bg7_1001 = p.loadImage(url_prefix + 'fourier7-0110.png');
    bg7_1000 = p.loadImage(url_prefix + 'fourier7-0111.png');
    bg7_0111 = p.loadImage(url_prefix + 'fourier7-1000.png');
    bg7_0110 = p.loadImage(url_prefix + 'fourier7-1001.png');
    bg7_0101 = p.loadImage(url_prefix + 'fourier7-1010.png');
    bg7_0100 = p.loadImage(url_prefix + 'fourier7-1011.png');
    bg7_0011 = p.loadImage(url_prefix + 'fourier7-1100.png');
    bg7_0010 = p.loadImage(url_prefix + 'fourier7-1101.png');
    bg7_0001 = p.loadImage(url_prefix + 'fourier7-1110.png');
    bg7_0000 = p.loadImage(url_prefix + 'fourier7-1111.png');
    
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

var myp57 = new p5(sketch7, 'sketch7');
///// Sketch 8

var sketch8 = function(p) {
    var mouse1 = 0, mouse2 = 0, mouse3 = 0, mouse4 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    
    bg8_1111 = p.loadImage(url_prefix + 'fourier8-0000.png');
    bg8_1110 = p.loadImage(url_prefix + 'fourier8-0001.png');
    bg8_1101 = p.loadImage(url_prefix + 'fourier8-0010.png');
    bg8_1100 = p.loadImage(url_prefix + 'fourier8-0011.png');
    bg8_1011 = p.loadImage(url_prefix + 'fourier8-0100.png');
    bg8_1010 = p.loadImage(url_prefix + 'fourier8-0101.png');
    bg8_1001 = p.loadImage(url_prefix + 'fourier8-0110.png');
    bg8_1000 = p.loadImage(url_prefix + 'fourier8-0111.png');
    bg8_0111 = p.loadImage(url_prefix + 'fourier8-1000.png');
    bg8_0110 = p.loadImage(url_prefix + 'fourier8-1001.png');
    bg8_0101 = p.loadImage(url_prefix + 'fourier8-1010.png');
    bg8_0100 = p.loadImage(url_prefix + 'fourier8-1011.png');
    bg8_0011 = p.loadImage(url_prefix + 'fourier8-1100.png');
    bg8_0010 = p.loadImage(url_prefix + 'fourier8-1101.png');
    bg8_0001 = p.loadImage(url_prefix + 'fourier8-1110.png');
    bg8_0000 = p.loadImage(url_prefix + 'fourier8-1111.png');
    
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

var myp58 = new p5(sketch8, 'sketch8');


///// Sketch 9

var sketch9 = function(p) {
    var mouse1 = 0, mouse2 = 0, mouse3 = 0, mouse4 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    
    bg9_1111 = p.loadImage(url_prefix + 'fourier9-0000.png');
    bg9_1110 = p.loadImage(url_prefix + 'fourier9-0001.png');
    bg9_1101 = p.loadImage(url_prefix + 'fourier9-0010.png');
    bg9_1100 = p.loadImage(url_prefix + 'fourier9-0011.png');
    bg9_1011 = p.loadImage(url_prefix + 'fourier9-0100.png');
    bg9_1010 = p.loadImage(url_prefix + 'fourier9-0101.png');
    bg9_1001 = p.loadImage(url_prefix + 'fourier9-0110.png');
    bg9_1000 = p.loadImage(url_prefix + 'fourier9-0111.png');
    bg9_0111 = p.loadImage(url_prefix + 'fourier9-1000.png');
    bg9_0110 = p.loadImage(url_prefix + 'fourier9-1001.png');
    bg9_0101 = p.loadImage(url_prefix + 'fourier9-1010.png');
    bg9_0100 = p.loadImage(url_prefix + 'fourier9-1011.png');
    bg9_0011 = p.loadImage(url_prefix + 'fourier9-1100.png');
    bg9_0010 = p.loadImage(url_prefix + 'fourier9-1101.png');
    bg9_0001 = p.loadImage(url_prefix + 'fourier9-1110.png');
    bg9_0000 = p.loadImage(url_prefix + 'fourier9-1111.png');
    
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

var myp59 = new p5(sketch9, 'sketch9');

///// Sketch 10

var sketch10 = function(p) {
    var mouse1 = 0, mouse2 = 0, mouse3 = 0, mouse4 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    
    bg10_1111 = p.loadImage(url_prefix + 'fourier10-0000.png');
    bg10_1110 = p.loadImage(url_prefix + 'fourier10-0001.png');
    bg10_1101 = p.loadImage(url_prefix + 'fourier10-0010.png');
    bg10_1100 = p.loadImage(url_prefix + 'fourier10-0011.png');
    bg10_1011 = p.loadImage(url_prefix + 'fourier10-0100.png');
    bg10_1010 = p.loadImage(url_prefix + 'fourier10-0101.png');
    bg10_1001 = p.loadImage(url_prefix + 'fourier10-0110.png');
    bg10_1000 = p.loadImage(url_prefix + 'fourier10-0111.png');
    bg10_0111 = p.loadImage(url_prefix + 'fourier10-1000.png');
    bg10_0110 = p.loadImage(url_prefix + 'fourier10-1001.png');
    bg10_0101 = p.loadImage(url_prefix + 'fourier10-1010.png');
    bg10_0100 = p.loadImage(url_prefix + 'fourier10-1011.png');
    bg10_0011 = p.loadImage(url_prefix + 'fourier10-1100.png');
    bg10_0010 = p.loadImage(url_prefix + 'fourier10-1101.png');
    bg10_0001 = p.loadImage(url_prefix + 'fourier10-1110.png');
    bg10_0000 = p.loadImage(url_prefix + 'fourier10-1111.png');
    
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

var myp510 = new p5(sketch10, 'sketch10');
///// Sketch 11

var sketch11 = function(p) {
    var mouse1 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    
    bg11_0 = p.loadImage(url_prefix + 'fourier11-0.png');
    bg11_1 = p.loadImage(url_prefix + 'fourier11-1.png');
    
    }

    p.draw = function() {
    str1 = mouse1.toString();
    picName = "bg11_" + str1;
    p.background(eval(picName));
    }

    p.mouseClicked = function() {
    if ((p.mouseX > 200) && (p.mouseX < 250) && (p.mouseY > 140) && (p.mouseY < 190)) {
        mouse1 = (mouse1 + 1)%2;
    }
    }
}

var myp511 = new p5(sketch11, 'sketch11');

///// Sketch 11_5

var sketch11_5 = function(p) {
    var mouse1 = 0;
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    
    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    
    bg11_2 = p.loadImage(url_prefix + 'fourier11-2.png');
    bg11_3 = p.loadImage(url_prefix + 'fourier11-3.png');
    
    }

    p.draw = function() {
    str1 = (mouse1+2).toString();
    picName = "bg11_" + str1;
    p.background(eval(picName));
    }

    p.mouseClicked = function() {
    if ((p.mouseX > 0) && (p.mouseX < width) && (p.mouseY > 0) && (p.mouseY < height)) {
        mouse1 = (mouse1 + 1)%2;
    }
    }
}

var myp5115 = new p5(sketch11_5, 'sketch11_5');

///// Sketch 12

var sketch12 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var theta = 0;
    var rad = 57;

    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    bg12 = p.loadImage('src/fourier12.png');
    label = p.loadImage('src/fourier12-diff.png');
    }

    p.draw = function() {
    p.background(bg12);

    if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
        theta = 2*3.15*p.mouseX/width;
    } 
    horz = p.cos(theta);
    vert = p.sin(theta);

    p.stroke(255, 181, 241);
    p.strokeWeight(2);
    p.line(222 + rad*horz, 153 - rad*vert, 222, 153);
    p.line(222 + rad*horz, 153 - rad*vert, 222, 153 - 2*rad*vert);
    p.stroke(199, 86, 178);
    p.line(222, 153, 222, 155 - 2*rad*vert);

    p.noStroke();
    p.fill(215, 162, 246);
    p.circle(p.mouseX, 255, 10);
    p.noFill();
    p.stroke(215, 162, 246);
    p.arc(222, 153, 30, 30, -theta, 0);

    p.image(label, 222 - 20, 153 - 2*rad*vert - 40, label.width/3.5, label.height/3.5);
    }
}

var myp512 = new p5(sketch12, 'sketch12');
///// Sketch 13

var sketch13 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var theta = 0;
    var rad = 43;

    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    bg13 = p.loadImage(url_prefix + 'fourier13.png');
    }

    p.draw = function() {
    p.background(bg13);

    if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
        theta = 2*2*3.155*p.mouseX/width - 0.02;
    } 
    horz = p.cos(theta);
    vert = p.sin(theta);
    horz2 = p.cos(2*theta);
    vert2 = p.sin(2*theta);
    horz3 = p.cos(3*theta);
    vert3 = p.sin(3*theta);

    p.stroke(255, 181, 241);
    p.strokeWeight(2);
    p.line(225 + rad*horz, 90 - rad*vert, 225, 90);
    p.line(225 + rad*horz, 90 - rad*vert, 225, 90 - 2*rad*vert);
    p.stroke(199, 86, 178);
    p.line(225, 90, 225, 90 - 2*rad*vert);

    if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
        p.noStroke();
        p.fill(215, 162, 246);
        p.circle(p.mouseX, 255, 10);
    }
    p.noFill();
    p.stroke(215, 162, 246);
    p.arc(225, 90, 30, 30, -theta, 0);

    p.stroke(255, 181, 241);
    p.strokeWeight(2);
    p.line(107 + rad*horz2, 162 - rad*vert2, 107, 162);
    p.line(107 + rad*horz2, 162 - rad*vert2, 107, 162 - 2*rad*vert2);
    p.stroke(199, 86, 178);
    p.line(107, 162, 107, 162 - 2*rad*vert2);

    p.noFill();
    p.stroke(215, 162, 246);
    p.arc(107, 162, 30, 30, -2*theta, 0);

    p.stroke(255, 181, 241);
    p.strokeWeight(2);
    p.line(331 + rad*horz3, 162 - rad*vert3, 331, 160);
    p.line(331 + rad*horz3, 162 - rad*vert3, 331, 160 - 2*rad*vert3);
    p.stroke(199, 86, 178);
    p.line(331, 162, 331, 162 - 2*rad*vert3);

    p.noFill();
    p.stroke(215, 162, 246);
    p.arc(331, 162, 30, 30, -3*theta, 0);
    }
}

var myp513 = new p5(sketch13, 'sketch13');

///// Sketch 14

var sketch14 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var theta = 0;
    var rad = 43;
    var angle1 = 0.02, angle2 = 0.13, angle3 = 0.4, angle4 = 0.63, angle5 = 0.7, angle6 = 0.95;

    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    bg13 = p.loadImage(url_prefix + 'fourier13.png');
    }

    p.draw = function() {
    p.background(bg13);

    if ((p.mouseY < height) && (p.mouseY > 0)) {
        if ((p.mouseX > 0) && (p.mouseX < angle1*width)) {
        theta = 2*2*3.155*angle1 - 0.02;
        }
        if ((p.mouseX > angle1*width) && (p.mouseX < angle2*width)) {
        theta = 2*2*3.155*angle2 - 0.02;
        }
        if ((p.mouseX > angle2*width) && (p.mouseX < angle3*width)) {
        theta = 2*2*3.155*angle3 - 0.02;
        }
        if ((p.mouseX > angle3*width) && (p.mouseX < angle4*width)) {
        theta = 2*2*3.155*angle4 - 0.02;
        }
        if ((p.mouseX > angle4*width) && (p.mouseX < angle5*width)) {
        theta = 2*2*3.155*angle5 - 0.02;
        }
        else if (p.mouseX > angle5*width) {
        theta = 2*2*3.155*angle6 - 0.02;
        }
    } 
    horz = p.cos(theta);
    vert = p.sin(theta);
    horz2 = p.cos(2*theta);
    vert2 = p.sin(2*theta);
    horz3 = p.cos(3*theta);
    vert3 = p.sin(3*theta);

    p.stroke(199, 86, 178);
    p.strokeWeight(2);
    p.line(225 + rad*horz, 90 - rad*vert, 225, 90);

    p.noStroke();
    p.fill(240, 238, 239);
    p.circle(angle1*width, 255, 10);
    p.circle(angle2*width, 255, 10);
    p.circle(angle3*width, 255, 10);
    p.circle(angle4*width, 255, 10);
    p.circle(angle5*width, 255, 10);
    p.circle(angle6*width, 255, 10);
    
    if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
        p.noStroke();
        p.fill(215, 162, 246);
        p.circle(theta*width/(2*2*3.155), 255, 10);
    }
    
    p.noFill();
    p.stroke(215, 162, 246);
    p.arc(225, 90, 30, 30, -theta, 0);

    p.stroke(199, 86, 178);
    p.strokeWeight(2);
    p.line(107 + rad*horz2, 162 - rad*vert2, 107, 162);

    p.noFill();
    p.stroke(215, 162, 246);
    p.arc(107, 162, 30, 30, -2*theta, 0);

    p.stroke(199, 86, 178);
    p.strokeWeight(2);
    p.line(331 + rad*horz3, 162 - rad*vert3, 331, 160);

    p.noFill();
    p.stroke(215, 162, 246);
    p.arc(331, 162, 30, 30, -3*theta, 0);
    }
}

var myp514 = new p5(sketch14, 'sketch14');

///// Sketch 14_5

var sketch14_5 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var rad = 43;
    var angle1 = 0.02, angle2 = 0.13, angle3 = 0.4, angle4 = 0.63, angle5 = 0.7, angle6 = 0.95;
    var theta = 0, angle = angle1;
    var xinit = 220, yinit = 153;
    var x = xinit, y = yinit;

    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    bg14_5 = p.loadImage(url_prefix + 'fourier14_5.png');
    }

    p.draw = function() {
    p.background(bg14_5);

    if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
        theta = 2*2*3.142*p.mouseX/width;
    } 

    p.stroke(255, 181, 241)
    p.strokeWeight(2);

    x = xinit;
    y = yinit;
    for (let i = -3; i < 4; i++) {
        p.line(x, y, x + rad*p.cos(i*(theta-angle)), y - rad*p.sin(i*(theta-angle)));
        x = x + rad*p.cos(i*(theta-angle));
        y = y - rad*p.sin(i*(theta-angle));
    }
    p.stroke(199, 86, 178);
    p.line(x, y, xinit, yinit);

    p.noStroke();
    p.fill(240, 238, 239);
    p.circle(angle1*width, 255, 10);
    p.circle(angle2*width, 255, 10);
    p.circle(angle3*width, 255, 10);
    p.circle(angle4*width, 255, 10);
    p.circle(angle5*width, 255, 10);
    p.circle(angle6*width, 255, 10);
    
    if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
        p.noStroke();
        p.fill(215, 162, 246);
        p.circle(theta*width/(2*2*3.155), 255, 10);
    }

    p.mouseClicked = function() {
        if ((p.mouseX > 0) && (p.mouseX < width) && (p.mouseY > 0) && (p.mouseY < height)) {
        if ((p.mouseY < height) && (p.mouseY > 0)) {
            if ((p.mouseX > 0) && (p.mouseX < angle1*width)) {
            angle = 2*2*3.142*angle1;
            }
            if ((p.mouseX > angle1*width) && (p.mouseX < angle2*width)) {
            angle = 2*2*3.142*angle2;
            }
            if ((p.mouseX > angle2*width) && (p.mouseX < angle3*width)) {
            angle = 2*2*3.142*angle3;
            }
            if ((p.mouseX > angle3*width) && (p.mouseX < angle4*width)) {
            angle = 2*2*3.142*angle4;
            }
            if ((p.mouseX > angle4*width) && (p.mouseX < angle5*width)) {
            angle = 2*2*3.142*angle5;
            }
            else if (p.mouseX > angle5*width) {
            angle = 2*2*3.142*angle6;
            }
        } 
        }
    }
        p.noStroke();
        p.fill(215, 162, 246);
        p.circle(angle*width/(2*2*3.155), 255, 10);
    }
}

var myp5 = new p5(sketch14_5, 'sketch14_5');

///// Sketch 15

var sketch15 = function(p) {
    var mod = 0.75;
    var width = 600*mod, height = 400*mod;
    var rad = 43;
    var angle1 = 1/7, angle2 = 2/7, angle3 = 3/7, angle4 = 4/7, angle5 = 5/7, angle6 = 6/7;
    var theta = 0, angle = 2*2*3.142*angle1;
    var xinit = 220, yinit = 153;
    var x = xinit, y = yinit;

    p.setup = function() {
    const canvas1 = p.createCanvas(width, height);
    bg14_5 = p.loadImage(url_prefix + 'fourier14_5.png');
    }

    p.draw = function() {
    p.background(bg14_5);

    if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
        theta = 2*2*3.14*p.mouseX/width;
    } 

    p.stroke(255, 181, 241)
    p.strokeWeight(2);

    x = xinit;
    y = yinit;
    for (let i = -3; i < 4; i++) {
        p.line(x, y, x + rad*p.cos(i*(theta-angle)), y - rad*p.sin(i*(theta-angle)));
        x = x + rad*p.cos(i*(theta-angle));
        y = y - rad*p.sin(i*(theta-angle));
    }
    p.stroke(199, 86, 178);
    p.line(x, y, xinit, yinit);

    p.noStroke();
    p.fill(240, 238, 239);
    p.circle(angle1*width, 255, 10);
    p.circle(angle2*width, 255, 10);
    p.circle(angle3*width, 255, 10);
    p.circle(angle4*width, 255, 10);
    p.circle(angle5*width, 255, 10);
    p.circle(angle6*width, 255, 10);
    
    if (((p.mouseY < height) && (p.mouseY > 0)) && ((p.mouseX < width) && (p.mouseX > 0))) {
        p.noStroke();
        p.fill(215, 162, 246);
        p.circle(theta*width/(2*2*3.155), 255, 10);
    }

    p.mouseClicked = function() {
        if ((p.mouseX > 0) && (p.mouseX < width) && (p.mouseY > 0) && (p.mouseY < height)) {
        if ((p.mouseY < height) && (p.mouseY > 0)) {
            if ((p.mouseX > 0) && (p.mouseX < angle1*width)) {
            angle = 2*2*3.142*angle1;
            }
            if ((p.mouseX > angle1*width) && (p.mouseX < angle2*width)) {
            angle = 2*2*3.142*angle2;
            }
            if ((p.mouseX > angle2*width) && (p.mouseX < angle3*width)) {
            angle = 2*2*3.142*angle3;
            }
            if ((p.mouseX > angle3*width) && (p.mouseX < angle4*width)) {
            angle = 2*2*3.142*angle4;
            }
            if ((p.mouseX > angle4*width) && (p.mouseX < angle5*width)) {
            angle = 2*2*3.142*angle5;
            }
            else if (p.mouseX > angle5*width) {
            angle = 2*2*3.142*angle6;
            }
        } 
        }
    }
        p.noStroke();
        p.fill(215, 162, 246);
        p.circle(angle*width/(2*2*3.155), 255, 10);
    }
}

var myp515 = new p5(sketch15, 'sketch15');
