package com.grievance.backend.service;
import com.grievance.backend.dto.ComplaintRequest;
import org.springframework.stereotype.Service;

@Service
public class ComplaintService {
    //@service anotaion says that this class as logic performance so create object and maintain it by yourself
    //@restcontroller receives the request from the frontend
    //@SpringBootApplication are saying that its an sprinboot application so spring scans and create objects
    public String receiveComplaint(ComplaintRequest complaintRequest){

        System.out.println("===== Complaint Details =====");
        System.out.println("Name : " + complaintRequest.getName());
        System.out.println("Email : " + complaintRequest.getEmail());
        System.out.println("Phone : " + complaintRequest.getPhone());
        System.out.println("District : " + complaintRequest.getDistrict());
        System.out.println("Address : " + complaintRequest.getAddress());
        System.out.println("Complaint Title : " + complaintRequest.getComplaintTitle());
        System.out.println("Complaint Description : " + complaintRequest.getComplaintDescription());

        return "Complaint received successfully!";
    }
}
