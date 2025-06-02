#include <SFML/Graphics.hpp>
#include "MLP.h"
using namespace sf;
#pragma GCC diagnostic ignored "-Wnarrowing"

int main()
{
    sf::RenderWindow window(sf::VideoMode({ 800, 800 }), "Deep Learning with Adam Optimizer");
    View view(FloatRect({ -1.2, -0.9 }, { 2.2, 2 }));
    window.setView(view);
    sf::CircleShape shape(0.1);
    shape.setFillColor(sf::Color::Green);

    const int nbpoints = 100;
    const int nbclasses = 3;
    const int nbInputParams = 2;


    dvector hidden_layers_sizes{ nbInputParams, 32, nbclasses };
    MLP NN(hidden_layers_sizes);


    // Train:
    int epochs = 1000;
    for (int i = 0; i < epochs; i++) {
        auto [x_train, y_test] = spiral_data(32, nbclasses, 0.05);
        dmatrix y_train = NN.forward(x_train);
        NN.backwards(x_train, y_test);
    }

    // Test:
    auto [x_train, y_test] = spiral_data(nbpoints, nbclasses, 0.05);
    dmatrix y_train = getCertitude(NN.forward(x_train));

    //print(y_train);
    //print(y_test);
    

    std::vector<CircleShape> points;
    int acc = 0;
    for (int i = 0; i < nbpoints * nbclasses; i++) {
        CircleShape point(0.01);
        if (y_test[i] == y_train[i]) {
            point.setFillColor(Color(static_cast<int>(255.f * (y_test[i][0])),
                static_cast<int>(255.f * (y_test[i][1])),
                static_cast<int>(255.f * (y_test[i][2]))));
            acc += 1;
        }
        else {
            point.setFillColor(Color(0.2 * 255, 0.2 * 255, 0.2 * 255));
        }

        point.setPosition({ x_train[i][0], x_train[i][1] });
        points.push_back(point);
    }

    print(100.f * acc / (nbpoints * nbclasses), "% d'accuracy");


    while (window.isOpen())
    {
        while (const std::optional event = window.pollEvent())
        {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (const auto* keyPressed = event->getIf<sf::Event::KeyPressed>())
                if (keyPressed->scancode == sf::Keyboard::Scancode::Escape)
                    window.close();
        }

        window.clear();
        for (int i = 0; i < points.size(); i++) window.draw(points[i]);
        window.display();
    }
}