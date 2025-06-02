#include <SFML/Graphics.hpp>
#include "Blocks.h"
using namespace sf;
#pragma GCC diagnostic ignored "-Wnarrowing"

int main()
{
    sf::RenderWindow window(sf::VideoMode({ 800, 800 }), "SFML works!");
    View view(FloatRect({ -1.2, -0.9 }, { 2.2, 2 }));
    window.setView(view);
    sf::CircleShape shape(0.1);
    shape.setFillColor(sf::Color::Green);

    const int nbpoints = 5;
    const int nbclasses = 3;

    const int nbInputParams = 2;
    
    auto [X, y] = spiral_data(nbpoints, nbclasses, 0.05);
    printArray(X, "Entrée");

    DL::Dense_layer layer(nbInputParams, 10);
    layer.forward(X);
    printArray(layer.output(), "Sortie du premier Layer");

    Activation activation_of_layer_1(layer.output());
    printArray(activation_of_layer_1.output(), "Sortie de la première activation");

    DL::Dense_layer layer2(10, nbclasses);
    layer2.forward(activation_of_layer_1.output());
    printArray(layer2.output(), "Sortie du 2e layer");

    Activation_Softmax activation_of_layer_2(layer2.output());
    printArray(activation_of_layer_2.output(), "Sortie du softmax");

    Cross_Entropy_Loss perte(activation_of_layer_2.output(), y);
    printArray(perte.getLoss().first);
    std::cout << perte.getLoss().second;


    std::vector<CircleShape> points;
    for (int i = 0; i < nbpoints * nbclasses; i++) {
        CircleShape point(0.01);
        point.setPosition({ X[i][0], X[i][1] });

        float color(255.f * (y[i] / 3.f));

        point.setFillColor(Color(static_cast<int>(255.f * (y[i] == 0 ? 1 : 0.2)),
                                 static_cast<int>(255.f * (y[i] == 1 ? 1 : 0.2)),
                                 static_cast<int>(255.f * (y[i] == 2 ? 1 : 0.2))));
        points.push_back(point);
    }

    std::cout << points.size();

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
        //window.draw(shape);
        for (int i = 0; i < points.size(); i++) {
            window.draw(points[i]);
        }
        window.display();
    }
}