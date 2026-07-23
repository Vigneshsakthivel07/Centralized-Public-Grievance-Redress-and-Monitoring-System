package com.grievance.backend.controller;
import com.grievance.backend.dto.ComplaintRequest;
import com.grievance.backend.service.ComplaintService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/complaints")
public class ComplaintController {
    @Autowired
    private ComplaintService complaintService;
    @PostMapping("/submit")
    public String submitComplaint(@RequestBody ComplaintRequest complaintRequest) {

        return complaintService.receiveComplaint(complaintRequest);

    }
}
