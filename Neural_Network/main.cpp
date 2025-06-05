#include <SFML/Graphics.hpp>
#include "TrainerClassifier/TrainerClassifier.h"
using namespace sf;
#pragma GCC diagnostic ignored "-Wnarrowing"


hyperparameters current_hyperparameters = {
    input_dim: 28 * 28,
    output_dim : 10,
    hidden_layers_sizes : {256, 128}, // Tested with { 64, 128, 256 } but same result.
    epochs : 0, // If 0, model will train on the whole MNIST database
    mini_batch_size : 32,
    learning_rate : 0.001,
    adam_beta_m : 0.9,
    adam_beta_v : 0.999,

    // Regularization
    dropout_rate : 0.001, // 0 if no dropout
    early_stopping : false,
    patience : 150,

    // Learn & tests
    learn : false,
    test : false,

    // Store loss and accuracy data in a .csv file for plotting
    store_data : false
};

int main() {
    // Ask for model training
    print("Train ? (y/n)"); char a; std::cin >> a;
    if (a == 'y') current_hyperparameters.learn = true;
    
    // MLP init
    MLP NN(current_hyperparameters);

    // Training
    if (current_hyperparameters.learn) {
        train(NN, current_hyperparameters);
        NN.saveWeights("model_weights.txt");
    }
    else {
        NN.loadWeights("model_weights.txt");
        print("Weights loaded !");
    }

    if (current_hyperparameters.test) test(NN, current_hyperparameters);

    // Window init
    sf::RenderWindow window(sf::VideoMode({ 800, 800 }), "Deep Learning with Adam Optimizer");
    window.setFramerateLimit(100);
    sf::View view({ 14, 14 }, {30, 30});
    window.setView(view);

    // Canvas init
    sf::RenderTexture canvas({ 28, 28 });
    canvas.clear(sf::Color::White);
    sf::Sprite sprite(canvas.getTexture());


    // Cursor init
    RectangleShape cursor({ 2, 2 });
    cursor.setFillColor(Color(255, 255, 255, 0));
    cursor.setOutlineThickness(0.5);
    cursor.setOrigin({ cursor.getSize().x / 2, cursor.getSize().y / 2 });
    cursor.setOutlineColor(Color(0, 0, 0));
    // Brush border with 20% opacity
    const float brush_size = 0.75;
    sf::CircleShape brush(brush_size*2, 5);
    brush.setOrigin({ brush_size*2, brush_size*2 });
    brush.setFillColor(Color(120, 0, 255, 50));
    // Brush center with 100% opacity
    sf::CircleShape brushCenter(brush_size, 5);
    brushCenter.setOrigin({ brush_size, brush_size });
    brushCenter.setFillColor(Color(120, 0, 255, 255));

    // Main loop
    bool firstPress = true;
    while (window.isOpen())
    {
        Vector2f mousePos = window.mapPixelToCoords(Mouse::getPosition(window));
        cursor.setPosition(mousePos);

        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>()) {
                // "Escape" closes the window
                if (keyPressed->scancode == sf::Keyboard::Scancode::Escape)
                    window.close();

                // "R" resets the canvas
                if (keyPressed->scancode == sf::Keyboard::Scancode::R) {
                    canvas.clear(sf::Color::White);
                    canvas.display();
                }

                // "A" gives number prediction from the canvas
                if (keyPressed->scancode == sf::Keyboard::Scancode::A) {
                    if (firstPress) {
                        dmatrix pixels(1, dvector(28 * 28, 0));
                        for (int i = 0; i < 28; i++)
                            for (int j = 0; j < 28; j++)
                                pixels[0][j * 28 + i] = 1.f - static_cast<int>(canvas.getTexture().copyToImage().getPixel({ i, j }).g) / 255.f;
                        
                        dmatrix output = NN.forward(pixels);
                        int number = std::distance(output[0].begin(), std::max_element(output[0].begin(), output[0].end()));
                        print("The number you've drawn is ", number, " !!!");
                    }
                    firstPress = false;
                }
            }
            else if (const auto* keyPressed = event->getIf<sf::Event::KeyReleased>())
                if (keyPressed->scancode == sf::Keyboard::Scancode::A)
                    firstPress = true;
        }

        // If I left click, it draws
        if (sf::Mouse::isButtonPressed(sf::Mouse::Button::Left)) {
            if (mousePos.x + 1 < 28 && mousePos.x > 1) {
                if (mousePos.y + 1 < 28 && mousePos.y > 1) {
                    brushCenter.setPosition(mousePos);
                    brush.setPosition(mousePos);
                    canvas.draw(brushCenter);
                    canvas.draw(brush);
                    canvas.display();
                }
            }
        }
         
        
        // Updating each frame
        window.clear(sf::Color(64, 64, 64));
        window.draw(sprite);
        window.draw(cursor);
        window.display();
    }
}