import React from "react";
import { Input } from "@shadcn/ui/input";
import { Textarea } from "@shadcn/ui/textarea";
import { Button } from "@shadcn/ui/button";
import { Label } from "@shadcn/ui/label";

const App = () => {
  return (
    <form className="p-4 max-w-md mx-auto">
      <Label htmlFor="name" className="block mb-1 font-semibold">
        Name
      </Label>
      <Input
        id="name"
        placeholder="Enter your name"
        className="w-full p-2 border rounded"
      />

      <Label htmlFor="message" className="block mt-4 mb-1 font-semibold">
        Message
      </Label>
      <Textarea
        id="message"
        rows={4}
        placeholder="Write your message"
        className="w-full p-2 border rounded"
      />

      <Button
        type="submit"
        className="mt-6 w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700"
      >
        Submit
      </Button>
    </form>
  );
};

export default App;
