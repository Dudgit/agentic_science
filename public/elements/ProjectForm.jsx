import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

const FIELDS = [
  ["project_name", "Project Name"],
  ["project_goal", "Project Objective & Core Goal"],
  ["tech_stack", "Target Tech Stack & Libraries"],
  ["constraints", "Deliverable Target & Constraints"],
]

export default function ProjectForm() {
  const [form, setForm] = useState({
    project_name: props.project_name,
    project_goal: props.project_goal,
    tech_stack: props.tech_stack,
    constraints: props.constraints,
  })

  return (
    <div className="flex flex-col gap-4 p-4 rounded-lg border">
      {FIELDS.map(([key, label]) => (
        <div key={key} className="flex flex-col gap-1">
          <Label>{label}</Label>
          <Input
            value={form[key]}
            onChange={(e) => setForm({ ...form, [key]: e.target.value })}
          />
        </div>
      ))}
      <div className="flex justify-end">
        <Button onClick={() => submitElement(form)}>Confirm &amp; Start</Button>
      </div>
    </div>
  )
}