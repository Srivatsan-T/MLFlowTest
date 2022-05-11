#include <iostream>
#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <string>
#include <string.h>

// Beginning of Linked list implementation

class list
{
    std::string data;
    list *next;

public:
    list(std::string new_data)
    {
        data = new_data;
        next = nullptr;
    }
    std::string get_data()
    {
        return data;
    }
    list *get_next()
    {
        return next;
    }
    void set_next(list *next_node)
    {
        this->next = next_node;
    }
};

list *head_node = nullptr;

int insert_list(std::string new_data)
{

    if (head_node == nullptr)
    {
        head_node = new list(new_data);
        return 1;
    }
    else
    {
        list *temp_iterator = head_node;
        while (temp_iterator->get_next() != nullptr)
        {
            temp_iterator = temp_iterator->get_next();
        }

        temp_iterator->set_next(new list(new_data));
        return 0;
    }
}

void print_list()
{
    if (head_node == nullptr)
    {
        std::cout << "The list is empty" << std::endl;
    }
    else
    {
        list *temp_iterator = head_node;
        while (temp_iterator != nullptr)
        {
            std::cout << temp_iterator->get_data() << std::endl;
            temp_iterator = temp_iterator->get_next();
        }
    }
}

// End of Linked list implementation

// Beginning of Socket Programming

int socket_id = 0;
int port_num;
struct sockaddr_in server_addr, client_addr;

void socket_creation()
{
    socket_id = socket(AF_INET, SOCK_STREAM, 0);
    if (socket_id < 0)
    {
        std::cout << "Error in opening a socket" << std::endl;
        exit(1);
    }
    std::cout<<"Socket created successfully"<<std::endl;
}

void bind_listen_and_accept(int port,int num_connections)
{
    port_num = port;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_num);

    int res = bind(socket_id, (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (res < 0)
    {
        std::cout << "Error in binding server" << std::endl;
        exit(1);
    }
    std::cout<<"Binding done successfully"<<std::endl;
    listen(socket_id, num_connections);
    socklen_t client_length = sizeof(client_addr);
    int client_socket_id = accept(socket_id, (struct sockaddr *)&client_addr, &client_length);
    if (client_socket_id < 0)
    {
        std::cout << "Error in accepting client" << std::endl;
        exit(1);
    }
    socket_id = client_socket_id;
    std::cout<<"Client accepted successfully"<<std::endl;
}

void connect_to(int port)
{
    port_num = port;
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = htons(port_num);

    int check = connect(socket_id, (struct sockaddr *)&server_addr, sizeof(server_addr));
    if (check < 0)
    {
        std::cout << "Error in connecting" << std::endl;
        exit(1);
    }
    std::cout<<"Connected to server successfully"<<std::endl;
}

char send_buffer[256];
char receive_buffer[256];

void send_message()
{
    std::cout<<"Enter the new-node content"<<std::endl;
    bzero(send_buffer,256);
    fgets(send_buffer,255,stdin);

    int length = strlen(send_buffer) -1;
    if(send_buffer[length] == '\n')
    {
        send_buffer[length] = '\0';
    }

    insert_list(send_buffer);
    int test = send(socket_id,send_buffer,255,0);
    if(test <0)
    {
        std::cout<<"Error in sending message"<<std::endl;
        exit(1);
    }
    std::cout<<"Message sent successfully"<<std::endl;
}


void receive_message()
{
    
    int test = recv(socket_id,receive_buffer,255,0);
    if(test <0)
    {
        std::cout<<"Error in receiving message"<<std::endl;
        exit(1);
    }
    insert_list((char*)receive_buffer);
    std::cout<<"Message received successfully"<<std::endl;
}



//Simulating 2 memebers in the network 

void first_instance()
{
    socket_creation();
    bind_listen_and_accept(6000,1);
    send_message();
    receive_message();
    print_list();
    shutdown(socket_id,2);
    std::cout<<"Connection is closed";
}

void second_instance()
{
    socket_creation();
    connect_to(6000);
    receive_message();
    send_message();
    print_list();
}

int main(int argc, char *argv[])
{

    print_list();
    insert_list("I");
    insert_list("am");
    insert_list("Srivatsan");
    print_list();
    if(argc == 2)
    {
        first_instance();
    }
    else
    {
        second_instance();
    }
    return 0;

}